#!/usr/bin/env python3
"""
Discord Bot for Lilith V2.

Uses the V2 CognitiveStage architecture with multi-tenant storage.

Features:
- Per-user isolated cognitive instances
- Per-server knowledge sharing (optional)
- Reaction-based feedback for learning
- Slash commands for teaching and control

Requirements:
    pip install discord.py python-dotenv

Setup:
    1. Create .env with DISCORD_TOKEN=your_token
    2. Run: python v2/discord_bot.py
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for discord.py
try:
    import discord
    from discord.ext import commands
    from discord import app_commands
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    discord = None
    commands = None
    app_commands = None

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


def _check_discord():
    if not DISCORD_AVAILABLE:
        raise RuntimeError(
            "discord.py is required. Install with: pip install discord.py"
        )


class LilithV2DiscordBot:
    """
    Discord bot adapter for Lilith V2 cognitive architecture.
    
    Each user gets their own CognitiveStage instance with isolated storage.
    """
    
    def __init__(
        self,
        data_path: str = "data",
        session_timeout_minutes: int = 30,
    ):
        _check_discord()
        
        from v2.lilith_v2.cognitive_stage import CognitiveStage
        from v2.lilith_v2.multi_tenant_store import MultiTenantPMFlowManager, MultiTenantGraphManager
        from v2.lilith_v2.relational_store import RelationalStore
        
        # Store class refs
        self._CognitiveStage = CognitiveStage
        self._MultiTenantPMFlowManager = MultiTenantPMFlowManager
        self._MultiTenantGraphManager = MultiTenantGraphManager
        self._RelationalStore = RelationalStore
        
        self.data_path = Path(data_path)
        self.session_timeout_minutes = session_timeout_minutes
        
        # Shared multi-tenant stores
        base_root = self.data_path / "base"
        users_root = self.data_path / "users"
        base_root.mkdir(parents=True, exist_ok=True)
        users_root.mkdir(parents=True, exist_ok=True)
        
        self.pmflow = MultiTenantPMFlowManager(str(base_root), str(users_root))
        self.graph = MultiTenantGraphManager(str(base_root), str(users_root))
        self.users_root = users_root
        
        # Create encoder
        self.encoder = self._create_encoder()
        
        # Discord bot setup
        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True
        
        self.bot = commands.Bot(
            command_prefix="!",
            intents=intents,
            description="Lilith V2 - A learning conversational AI"
        )
        
        # Per-user cognitive instances
        self._user_stages: Dict[str, CognitiveStage] = {}
        self._user_last_active: Dict[str, datetime] = {}
        
        # Track messages for reaction feedback (message_id -> user_id)
        self._message_users: Dict[int, str] = {}
        
        # User display names
        self._user_names: Dict[str, str] = {}
        
        self._setup_events()
        self._setup_commands()
    
    def _create_encoder(self):
        """Create the best available encoder."""
        try:
            from pmflow import PMFlowEmbeddingEncoder
            return PMFlowEmbeddingEncoder(
                dimension=96,
                latent_dim=48,
                enable_flow=True,
            )
        except ImportError:
            # Fallback
            import torch
            class SimpleEncoder:
                def __init__(self):
                    self.embedding_dim = 64
                    self.enable_flow = False
                def encode(self, text):
                    seed = sum(ord(c) for c in str(text))
                    torch.manual_seed(seed)
                    return torch.randn(64)
            return SimpleEncoder()
    
    def _get_user_id(self, discord_user) -> str:
        """Convert Discord user to Lilith user ID."""
        return f"discord_{discord_user.id}"
    
    def _get_or_create_stage(self, discord_user, guild_id: Optional[str] = None):
        """Get or create a CognitiveStage for a user."""
        user_id = self._get_user_id(discord_user)
        cache_key = f"{user_id}:{guild_id or 'dm'}"
        
        if cache_key in self._user_stages:
            self._user_last_active[cache_key] = datetime.now()
            return self._user_stages[cache_key]
        
        # Create response store for this user
        user_path = self.users_root / user_id
        user_path.mkdir(parents=True, exist_ok=True)
        response_store = self._RelationalStore(str(user_path / "responses.db"))
        
        # Create cognitive stage
        stage = self._CognitiveStage(
            node_id=cache_key,
            pmflow_store=self.pmflow,
            graph_store=self.graph,
            encoder=self.encoder,
            config={
                "knowledge_enabled": True,
                "response_store": response_store,
                "composition_mode": "adaptive",
                "enable_blending": True,
                "enable_learning": True,
                "enable_reasoning": True,
                "deliberation_steps": 10,
            }
        )
        
        self._user_stages[cache_key] = stage
        self._user_last_active[cache_key] = datetime.now()
        self._user_names[user_id] = discord_user.display_name
        
        logger.info(f"Created cognitive stage for {cache_key}")
        return stage
    
    def _cleanup_inactive_sessions(self) -> int:
        """Free memory for inactive sessions."""
        if self.session_timeout_minutes <= 0:
            return 0
        
        threshold = datetime.now() - timedelta(minutes=self.session_timeout_minutes)
        to_remove = [
            key for key, last_active in self._user_last_active.items()
            if last_active < threshold
        ]
        
        for key in to_remove:
            if key in self._user_stages:
                del self._user_stages[key]
            self._user_last_active.pop(key, None)
            logger.info(f"Cleaned up inactive session: {key}")
        
        return len(to_remove)
    
    def _setup_events(self):
        """Setup Discord event handlers."""
        
        @self.bot.event
        async def on_ready():
            print(f"\n{'='*50}")
            print(f"🌙 Lilith V2 Discord Bot Online")
            print(f"   Logged in as: {self.bot.user}")
            print(f"   Servers: {len(self.bot.guilds)}")
            print(f"{'='*50}\n")
            
            # Sync slash commands
            try:
                synced = await self.bot.tree.sync()
                print(f"   Synced {len(synced)} slash commands")
            except Exception as e:
                print(f"   Failed to sync commands: {e}")
            
            # Start cleanup task
            self.bot.loop.create_task(self._background_cleanup())
        
        @self.bot.event
        async def on_message(message):
            # Ignore bots
            if message.author.bot:
                return
            
            # Process commands first
            await self.bot.process_commands(message)
            
            # Check if we should respond
            should_respond = False
            guild_id = str(message.guild.id) if message.guild else None
            
            # DMs: always respond
            if message.guild is None:
                should_respond = True
            # Mentions: respond
            elif self.bot.user in message.mentions:
                should_respond = True
            # Reply to bot: respond
            elif message.reference and message.reference.resolved:
                if message.reference.resolved.author == self.bot.user:
                    should_respond = True
            
            if should_respond:
                # Remove bot mention from content
                content = message.content
                if self.bot.user:
                    content = content.replace(f'<@{self.bot.user.id}>', '').strip()
                    content = content.replace(f'<@!{self.bot.user.id}>', '').strip()
                
                if not content:
                    return
                
                async with message.channel.typing():
                    response = await self._process_message(
                        message.author,
                        content,
                        message.id,
                        guild_id=guild_id
                    )
                
                if response:
                    # Split long responses
                    if len(response) > 2000:
                        chunks = [response[i:i+1990] for i in range(0, len(response), 1990)]
                        for chunk in chunks:
                            sent = await message.reply(chunk)
                            self._message_users[sent.id] = self._get_user_id(message.author)
                    else:
                        sent = await message.reply(response)
                        self._message_users[sent.id] = self._get_user_id(message.author)
        
        @self.bot.event
        async def on_reaction_add(reaction, user):
            """Handle reaction feedback."""
            if user.bot:
                return
            
            message_id = reaction.message.id
            if message_id not in self._message_users:
                return
            
            # Check reaction is from original user
            stored_user_id = self._message_users[message_id]
            if self._get_user_id(user) != stored_user_id:
                return
            
            # Find the stage
            guild_id = str(reaction.message.guild.id) if reaction.message.guild else None
            cache_key = f"{stored_user_id}:{guild_id or 'dm'}"
            stage = self._user_stages.get(cache_key)
            
            if stage:
                emoji = str(reaction.emoji)
                if emoji in ['👍', '❤️', '✅', '🎉', '💯', '👏']:
                    stage.record_response_outcome(success=True)
                    logger.debug(f"Positive feedback from {stored_user_id}")
                elif emoji in ['👎', '❌', '😕', '🤔', '😐']:
                    stage.record_response_outcome(success=False)
                    logger.debug(f"Negative feedback from {stored_user_id}")
    
    async def _process_message(
        self,
        user,
        content: str,
        message_id: int,
        guild_id: Optional[str] = None
    ) -> Optional[str]:
        """Process a message and generate response."""
        try:
            stage = self._get_or_create_stage(user, guild_id)
            user_id = self._get_user_id(user)
            
            # Get context for multi-tenant
            ctx = {
                "tenant": user_id,
                "guild": guild_id,
                "modality": "text",
            }
            
            # Process through cognitive stage
            stage.learn(content, ctx)
            
            # Get response from last thought
            if hasattr(stage, 'last_thought') and stage.last_thought:
                response = stage.last_thought.get("response", "")
                if response:
                    return response
            
            return "I'm still learning. Could you tell me more?"
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()
            return "I encountered an error processing that. Please try again."
    
    async def _background_cleanup(self):
        """Periodically cleanup inactive sessions."""
        await self.bot.wait_until_ready()
        
        while not self.bot.is_closed():
            await asyncio.sleep(300)  # Every 5 minutes
            try:
                cleaned = self._cleanup_inactive_sessions()
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} inactive sessions")
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    def _setup_commands(self):
        """Setup slash commands."""
        
        @self.bot.tree.command(name="help", description="Show Lilith help")
        async def help_cmd(interaction):
            embed = discord.Embed(
                title="🌙 Lilith V2 Help",
                description="I'm a learning AI. Talk to me and I'll remember!",
                color=0x9B59B6
            )
            embed.add_field(
                name="💬 Chatting",
                value="• Mention me or reply to my messages\n• I learn from our conversations",
                inline=False
            )
            embed.add_field(
                name="👍 Feedback",
                value="• React 👍 or 👎 to my responses\n• This helps me learn what works",
                inline=False
            )
            embed.add_field(
                name="📚 Commands",
                value="• `/teach` - Teach me a fact\n• `/stats` - My learning stats\n• `/forget` - Clear my memory of you",
                inline=False
            )
            await interaction.response.send_message(embed=embed)
        
        @self.bot.tree.command(name="stats", description="Show Lilith statistics")
        async def stats_cmd(interaction):
            user_id = self._get_user_id(interaction.user)
            guild_id = str(interaction.guild.id) if interaction.guild else None
            cache_key = f"{user_id}:{guild_id or 'dm'}"
            
            stage = self._user_stages.get(cache_key)
            
            embed = discord.Embed(
                title="📊 Lilith V2 Stats",
                color=0x3498DB
            )
            
            if stage and hasattr(stage, 'stats'):
                stats = stage.stats()
                embed.add_field(name="Active Sessions", value=str(len(self._user_stages)), inline=True)
                embed.add_field(name="Your Session", value="Active" if stage else "Inactive", inline=True)
                
                # Add cognitive stats if available
                if 'attractors' in stats:
                    embed.add_field(name="Learned Concepts", value=str(stats['attractors']), inline=True)
                if 'response_patterns' in stats:
                    embed.add_field(name="Response Patterns", value=str(stats['response_patterns']), inline=True)
            else:
                embed.add_field(name="Active Sessions", value=str(len(self._user_stages)), inline=True)
                embed.add_field(name="Your Session", value="Not started yet", inline=True)
            
            await interaction.response.send_message(embed=embed)
        
        @self.bot.tree.command(name="teach", description="Teach Lilith something new")
        @app_commands.describe(
            fact="What you want to teach (e.g., 'Dogs are mammals')"
        )
        async def teach_cmd(interaction, fact: str):
            user_id = self._get_user_id(interaction.user)
            guild_id = str(interaction.guild.id) if interaction.guild else None
            
            stage = self._get_or_create_stage(interaction.user, guild_id)
            
            ctx = {
                "tenant": user_id,
                "guild": guild_id,
                "teaching": True,  # Hint for lower extraction threshold
            }
            
            # Process as a teaching interaction
            stage.learn(fact, ctx)
            
            # Check what was learned
            learned = []
            if hasattr(stage, 'last_thought') and stage.last_thought:
                extracted = stage.last_thought.get("extracted_knowledge", [])
                for rel in extracted:
                    learned.append(f"• {rel.subject} {rel.predicate} {rel.object}")
            
            if learned:
                response = "📝 I've learned:\n" + "\n".join(learned)
            else:
                response = "🤔 I tried to learn from that, but couldn't extract a clear fact. Try stating it as 'X is Y' or 'X are Y'."
            
            await interaction.response.send_message(response)
        
        @self.bot.tree.command(name="forget", description="Clear your conversation history")
        async def forget_cmd(interaction):
            user_id = self._get_user_id(interaction.user)
            guild_id = str(interaction.guild.id) if interaction.guild else None
            cache_key = f"{user_id}:{guild_id or 'dm'}"
            
            # Remove stage from cache (data on disk remains)
            if cache_key in self._user_stages:
                del self._user_stages[cache_key]
                self._user_last_active.pop(cache_key, None)
            
            await interaction.response.send_message(
                "🧹 I've cleared my active memory of our conversation. "
                "Note: Learned facts remain in my knowledge base."
            )
        
        @self.bot.tree.command(name="reasoning", description="Show Lilith's last reasoning chain")
        async def reasoning_cmd(interaction):
            user_id = self._get_user_id(interaction.user)
            guild_id = str(interaction.guild.id) if interaction.guild else None
            cache_key = f"{user_id}:{guild_id or 'dm'}"
            
            stage = self._user_stages.get(cache_key)
            
            if not stage or not hasattr(stage, '_reasoning') or not stage._reasoning:
                await interaction.response.send_message("No reasoning data available yet. Chat with me first!")
                return
            
            result = stage._reasoning.get_last_result()
            if not result:
                await interaction.response.send_message("No reasoning result available.")
                return
            
            embed = discord.Embed(
                title="🧠 Last Reasoning Chain",
                color=0xE74C3C
            )
            embed.add_field(name="Steps", value=str(result.deliberation_steps), inline=True)
            embed.add_field(name="Confidence", value=f"{result.confidence:.2f}", inline=True)
            embed.add_field(name="Focus", value=result.focus_concept or "None", inline=True)
            embed.add_field(name="Intent", value=result.resolved_intent or "None", inline=True)
            
            if result.inferences:
                inf_text = "\n".join(f"• {inf.conclusion[:50]}..." for inf in result.inferences[:5])
                embed.add_field(name="Inferences", value=inf_text, inline=False)
            
            await interaction.response.send_message(embed=embed)
    
    def run(self, token: Optional[str] = None):
        """Run the Discord bot."""
        token = token or os.getenv("DISCORD_TOKEN")
        
        if not token:
            print("Error: No Discord token provided.")
            print("Set DISCORD_TOKEN environment variable or create a .env file.")
            sys.exit(1)
        
        print("🌙 Starting Lilith V2 Discord Bot...")
        self.bot.run(token)
    
    def close(self):
        """Cleanup resources."""
        self.pmflow.close()
        self.graph.close()


def main():
    import argparse
    
    if not DISCORD_AVAILABLE:
        print("Error: discord.py is required")
        print("Install with: pip install discord.py")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Lilith V2 Discord Bot")
    parser.add_argument("--timeout", "-t", type=int, default=30,
                        help="Session timeout in minutes (default: 30)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    bot = LilithV2DiscordBot(session_timeout_minutes=args.timeout)
    
    try:
        bot.run()
    finally:
        bot.close()


if __name__ == "__main__":
    main()
