"""
Discord Bot Adapter for Lilith AI.

This module provides a Discord interface for Lilith, allowing users to
interact with the AI through Discord messages and commands.

Features:
- User isolation via Discord user IDs
- Automatic user name learning from conversation
- Explicit name setting via /setname command
- Per-channel or DM conversations
- Auto-feedback detection from Discord reactions

Requirements:
    pip install discord.py python-dotenv

Usage:
    1. Create a .env file with DISCORD_TOKEN=your_bot_token
    2. Run: python discord_bot.py
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Check for discord.py
try:
    import discord
    from discord.ext import commands
    from discord import app_commands
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    discord = None  # type: ignore
    commands = None  # type: ignore
    app_commands = None  # type: ignore

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


def _check_discord():
    """Raise error if discord.py not available."""
    if not DISCORD_AVAILABLE:
        raise RuntimeError(
            "discord.py is required for the Discord bot. "
            "Install with: pip install discord.py"
        )


class LilithDiscordBot:
    """
    Discord bot adapter for Lilith AI.
    
    Manages per-user instances of Lilith components with isolated storage.
    """
    
    def __init__(self, 
                 data_path: str = "data",
                 command_prefix: str = "!",
                 session_timeout_minutes: int = 30,
                 user_retention_days: float = 7.0):
        """
        Initialize the Discord bot.
        
        Args:
            data_path: Base path for data storage
            command_prefix: Prefix for legacy commands (slash commands recommended)
            session_timeout_minutes: Minutes of inactivity before freeing session memory
            user_retention_days: Days of inactivity before deleting user data (0 = never)
        """
        _check_discord()
        
        # Import Lilith components here to avoid import errors when discord not available
        from lilith.embedding import PMFlowEmbeddingEncoder
        from lilith.response_composer import ResponseComposer
        from lilith.conversation_state import ConversationState
        from lilith.multi_tenant_store import MultiTenantFragmentStore
        from lilith.user_auth import AuthMode, UserIdentity
        from lilith.user_preferences import UserPreferenceLearner, UserPreferencesStore
        
        self.data_path = Path(data_path)
        self.session_timeout_minutes = session_timeout_minutes
        self.user_retention_days = user_retention_days
        
        # Store class references for later use
        self._ResponseComposer = ResponseComposer
        self._ConversationState = ConversationState
        self._MultiTenantFragmentStore = MultiTenantFragmentStore
        self._AuthMode = AuthMode
        self._UserIdentity = UserIdentity
        
        # Discord bot setup with intents
        intents = discord.Intents.default()
        intents.message_content = True  # Required to read message content
        intents.reactions = True  # For reaction-based feedback
        
        self.bot = commands.Bot(
            command_prefix=command_prefix,
            intents=intents,
            description="Lilith - A learning conversational AI"
        )
        
        # Shared components (read-only)
        self.encoder = PMFlowEmbeddingEncoder()
        self.preference_store = UserPreferencesStore(str(self.data_path))
        self.preference_learner = UserPreferenceLearner(store=self.preference_store)
        
        # Optional: Auto semantic learning
        self._AutoSemanticLearner = None
        try:
            from lilith.auto_semantic_learner import AutoSemanticLearner
            self._AutoSemanticLearner = AutoSemanticLearner
        except ImportError:
            pass
        
        # Optional: Feedback detection
        self._FeedbackTracker = None
        try:
            from lilith.feedback_detector import FeedbackTracker
            self._FeedbackTracker = FeedbackTracker
        except ImportError:
            pass
        
        # Per-user state caches with last activity tracking
        self._user_composers: Dict[str, Any] = {}
        self._user_stores: Dict[str, Any] = {}
        self._user_states: Dict[str, Any] = {}
        self._user_trackers: Dict[str, Any] = {}
        self._user_auto_learners: Dict[str, Any] = {}
        self._user_last_active: Dict[str, datetime] = {}  # Track session activity
        
        # Track last messages for feedback (message_id -> (user_id, pattern_id))
        self._message_patterns: Dict[int, tuple] = {}
        
        # Setup event handlers
        self._setup_events()
        self._setup_commands()
    
    def _touch_session(self, user_id: str) -> None:
        """Update session activity timestamp."""
        from datetime import datetime
        self._user_last_active[user_id] = datetime.now()
    
    def _cleanup_inactive_sessions(self) -> int:
        """
        Free memory for sessions inactive longer than timeout.
        
        Returns:
            Number of sessions cleaned up
        """
        from datetime import datetime, timedelta
        
        if self.session_timeout_minutes <= 0:
            return 0
        
        threshold = datetime.now() - timedelta(minutes=self.session_timeout_minutes)
        to_cleanup = []
        
        for user_id, last_active in self._user_last_active.items():
            if last_active < threshold:
                to_cleanup.append(user_id)
        
        for user_id in to_cleanup:
            self._free_user_session(user_id)
        
        return len(to_cleanup)
    
    def _free_user_session(self, user_id: str) -> None:
        """
        Free memory for a user's session (keeps data on disk).
        
        Args:
            user_id: User whose session to free
        """
        # Remove from all caches
        self._user_composers.pop(user_id, None)
        self._user_stores.pop(user_id, None)
        self._user_states.pop(user_id, None)
        self._user_trackers.pop(user_id, None)
        self._user_auto_learners.pop(user_id, None)
        self._user_last_active.pop(user_id, None)
        
        # Clear from preference cache too
        self.preference_store._cache.pop(user_id, None)
    
    def _cleanup_old_user_data(self) -> List[str]:
        """
        Delete data for users inactive longer than retention period.
        
        Returns:
            List of deleted user IDs
        """
        if self.user_retention_days <= 0:
            return []
        
        # First free their sessions
        inactive = self.preference_store.get_inactive_users(self.user_retention_days)
        for user_id, _ in inactive:
            self._free_user_session(user_id)
        
        # Then delete their data
        return self.preference_store.cleanup_inactive_users(
            days=self.user_retention_days, 
            dry_run=False
        )
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about active sessions."""
        from datetime import datetime
        
        active_sessions = len(self._user_composers)
        
        # Find oldest and newest sessions
        oldest_age = 0.0
        newest_age = float('inf')
        
        for user_id, last_active in self._user_last_active.items():
            age_minutes = (datetime.now() - last_active).total_seconds() / 60
            oldest_age = max(oldest_age, age_minutes)
            newest_age = min(newest_age, age_minutes)
        
        return {
            'active_sessions': active_sessions,
            'oldest_session_minutes': oldest_age if active_sessions > 0 else 0,
            'newest_session_minutes': newest_age if active_sessions > 0 else 0,
            'session_timeout_minutes': self.session_timeout_minutes,
            'user_retention_days': self.user_retention_days,
        }
    
    async def _background_cleanup(self):
        """Background task that periodically cleans up inactive sessions and old user data."""
        import asyncio
        
        while True:
            try:
                # Run cleanup every 5 minutes
                await asyncio.sleep(300)
                
                # Cleanup inactive sessions
                freed_count = self._cleanup_inactive_sessions()
                if freed_count > 0:
                    print(f"🧹 Freed {freed_count} inactive sessions")
                
                # Cleanup old user data (check once per hour, but we run every 5 min so use counter)
                if not hasattr(self, '_cleanup_counter'):
                    self._cleanup_counter = 0
                self._cleanup_counter += 1
                
                if self._cleanup_counter >= 12:  # 12 * 5 min = 1 hour
                    self._cleanup_counter = 0
                    deleted_users = self._cleanup_old_user_data()
                    if deleted_users:
                        print(f"🗑️ Deleted data for {len(deleted_users)} inactive users: {deleted_users}")
                        
            except Exception as e:
                print(f"⚠️ Background cleanup error: {e}")
                continue

    
    def _get_user_id(self, discord_user) -> str:
        """Convert Discord user to Lilith user ID."""
        return f"discord_{discord_user.id}"
    
    def _get_or_create_composer(self, discord_user):
        """Get or create a ResponseComposer for a user."""
        user_id = self._get_user_id(discord_user)
        
        if user_id in self._user_composers:
            return self._user_composers[user_id]
        
        # Create user identity
        identity = self._UserIdentity(
            user_id=user_id,
            auth_mode=self._AuthMode.TRUSTED,
            display_name=discord_user.display_name
        )
        
        # Create fragment store for this user
        fragment_store = self._MultiTenantFragmentStore(
            encoder=self.encoder,
            user_identity=identity,
            base_data_path=str(self.data_path)
        )
        
        # Create conversation state
        state = self._ConversationState(self.encoder)
        
        # Create response composer
        composer = self._ResponseComposer(
            fragment_store,
            state,
            semantic_encoder=self.encoder,
            enable_knowledge_augmentation=True,
            enable_modal_routing=True
        )
        
        # Load contrastive weights if available
        contrastive_path = self.data_path / "contrastive_learner"
        if contrastive_path.with_suffix('.json').exists():
            composer.load_contrastive_weights(str(contrastive_path))
        
        # Create feedback tracker
        if self._FeedbackTracker:
            tracker = self._FeedbackTracker()
            self._user_trackers[user_id] = tracker
        
        # Create auto learner
        if self._AutoSemanticLearner and composer.contrastive_learner:
            auto_learner = self._AutoSemanticLearner(
                contrastive_learner=composer.contrastive_learner,
                auto_train_threshold=10,
                auto_train_steps=3
            )
            self._user_auto_learners[user_id] = auto_learner
        
        # Cache all components
        self._user_composers[user_id] = composer
        self._user_stores[user_id] = fragment_store
        self._user_states[user_id] = state
        
        return composer
    
    def _setup_events(self):
        """Setup Discord event handlers."""
        
        @self.bot.event
        async def on_ready():
            """Called when bot is ready."""
            print(f"🤖 Lilith Discord Bot is ready!")
            print(f"   Logged in as: {self.bot.user}")
            print(f"   Servers: {len(self.bot.guilds)}")
            print(f"   Session timeout: {self.session_timeout_minutes} minutes")
            print(f"   User retention: {self.user_retention_days} days")
            
            # Sync slash commands
            try:
                synced = await self.bot.tree.sync()
                print(f"   Synced {len(synced)} slash commands")
            except Exception as e:
                print(f"   Failed to sync commands: {e}")
            
            # Start background cleanup task
            self.bot.loop.create_task(self._background_cleanup())
        
        @self.bot.event
        async def on_message(message):
            """Handle incoming messages."""
            # Ignore messages from the bot itself
            if message.author == self.bot.user:
                return
            
            # Ignore messages that are commands
            if message.content.startswith(self.bot.command_prefix):
                await self.bot.process_commands(message)
                return
            
            # Only respond to DMs or when mentioned
            is_dm = isinstance(message.channel, discord.DMChannel)
            is_mentioned = self.bot.user in message.mentions
            
            if not (is_dm or is_mentioned):
                return
            
            # Remove mention from message
            content = message.content
            if is_mentioned:
                content = content.replace(f'<@{self.bot.user.id}>', '').strip()
                content = content.replace(f'<@!{self.bot.user.id}>', '').strip()
            
            if not content:
                return
            
            # Process the message
            async with message.channel.typing():
                response_text = await self._process_message(
                    message.author, 
                    content,
                    message.id
                )
            
            # Send response
            if response_text:
                sent_message = await message.reply(response_text)
                
                # Track for reaction feedback
                user_id = self._get_user_id(message.author)
                if user_id in self._user_trackers:
                    tracker = self._user_trackers[user_id]
                    if tracker.history:
                        last = tracker.history[-1]
                        if last.get('pattern_id'):
                            self._message_patterns[sent_message.id] = (
                                user_id, 
                                last['pattern_id']
                            )
        
        @self.bot.event
        async def on_reaction_add(reaction, user):
            """Handle reactions for feedback."""
            # Ignore bot's own reactions
            if user == self.bot.user:
                return
            
            # Check if this is a Lilith response
            message_id = reaction.message.id
            if message_id not in self._message_patterns:
                return
            
            user_id, pattern_id = self._message_patterns[message_id]
            
            # Check if reaction is from the original user
            if self._get_user_id(user) != user_id:
                return
            
            # Apply feedback based on emoji
            emoji = str(reaction.emoji)
            store = self._user_stores.get(user_id)
            
            if store:
                if emoji in ['👍', '❤️', '✅', '🎉', '💯']:
                    store.upvote(pattern_id, strength=0.2)
                    print(f"   👍 Upvoted {pattern_id} via reaction")
                elif emoji in ['👎', '❌', '😕', '🤔']:
                    store.downvote(pattern_id, strength=0.2)
                    print(f"   👎 Downvoted {pattern_id} via reaction")
    
    async def _process_message(self, user, content: str, message_id: int) -> str:
        """
        Process a message and generate a response.
        
        Args:
            user: Discord user
            content: Message content
            message_id: Message ID for tracking
            
        Returns:
            Response text
        """
        from datetime import datetime
        
        user_id = self._get_user_id(user)
        
        # Track user activity for session management
        self._user_last_active[user_id] = datetime.now()
        self.preference_store.touch_user(user_id)
        
        # Get or create composer for this user
        composer = self._get_or_create_composer(user)
        
        # Check for preference information (name, interests)
        learned = self.preference_learner.process_input(user_id, content)
        
        # Check for feedback signals from previous message
        tracker = self._user_trackers.get(user_id)
        store = self._user_stores.get(user_id)
        
        if tracker and store and tracker.history:
            # Get last response info
            feedback_result = tracker.check_feedback(content)
            
            if feedback_result:
                result, pattern_id = feedback_result
                if result.should_apply and pattern_id:
                    if result.is_positive:
                        store.upvote(pattern_id, strength=result.strength)
                    elif result.is_negative:
                        store.downvote(pattern_id, strength=result.strength)
        
        # Generate response
        response = composer.compose_response(context=content, user_input=content)
        
        # Track for feedback
        if tracker:
            pattern_id = response.fragment_ids[0] if response.fragment_ids else None
            tracker.record_interaction(content, response.text, pattern_id)
        
        # Auto-learn semantic relationships
        auto_learner = self._user_auto_learners.get(user_id)
        if auto_learner:
            auto_learner.process_conversation(content, response.text)
        
        # Build response text
        response_text = response.text
        
        # If we learned their name, add confirmation
        if 'name' in learned:
            name = learned['name']
            confirmation = self.preference_learner.format_name_confirmation(name)
            response_text = f"{confirmation}\n\n{response_text}"
        
        return response_text
    
    def _setup_commands(self):
        """Setup slash commands."""
        
        @self.bot.tree.command(name="setname", description="Set your preferred name for Lilith to use")
        @app_commands.describe(name="Your preferred name")
        async def setname(interaction, name: str):
            """Set user's preferred name."""
            user_id = self._get_user_id(interaction.user)
            
            # Validate name
            if len(name) < 1 or len(name) > 50:
                await interaction.response.send_message(
                    "Name must be between 1 and 50 characters.",
                    ephemeral=True
                )
                return
            
            # Set the name
            self.preference_learner.set_user_name(user_id, name)
            
            await interaction.response.send_message(
                f"Got it! I'll call you **{name}** from now on. 😊",
                ephemeral=False
            )
        
        @self.bot.tree.command(name="whoami", description="Check what Lilith knows about you")
        async def whoami(interaction):
            """Show what Lilith knows about the user."""
            user_id = self._get_user_id(interaction.user)
            prefs = self.preference_store.load(user_id)
            
            embed = discord.Embed(
                title="What I Know About You",
                color=discord.Color.purple()
            )
            
            embed.add_field(
                name="Your Name",
                value=prefs.display_name or "Not set (tell me or use /setname)",
                inline=False
            )
            
            if prefs.interests:
                embed.add_field(
                    name="Your Interests",
                    value=", ".join(prefs.interests[:10]),
                    inline=False
                )
            
            # Get pattern stats
            store = self._user_stores.get(user_id)
            if store:
                counts = store.get_pattern_count()
                embed.add_field(
                    name="Your Learned Patterns",
                    value=str(counts.get('user', 0)),
                    inline=True
                )
                embed.add_field(
                    name="Base Knowledge",
                    value=str(counts.get('base', 0)),
                    inline=True
                )
            
            embed.set_footer(text=f"User ID: {user_id}")
            
            await interaction.response.send_message(embed=embed, ephemeral=True)
        
        @self.bot.tree.command(name="stats", description="Show Lilith statistics")
        async def stats(interaction):
            """Show bot statistics."""
            user_id = self._get_user_id(interaction.user)
            
            embed = discord.Embed(
                title="📊 Lilith Statistics",
                color=discord.Color.blue()
            )
            
            embed.add_field(
                name="Active Users",
                value=str(len(self._user_composers)),
                inline=True
            )
            
            embed.add_field(
                name="Servers",
                value=str(len(self.bot.guilds)),
                inline=True
            )
            
            # User-specific stats
            tracker = self._user_trackers.get(user_id)
            if tracker:
                fb_stats = tracker.get_stats()
                embed.add_field(
                    name="Your Interactions",
                    value=str(fb_stats['total_interactions']),
                    inline=True
                )
                if fb_stats['total_interactions'] > 0:
                    embed.add_field(
                        name="Positive Feedback Rate",
                        value=f"{fb_stats['positive_rate']:.0%}",
                        inline=True
                    )
            
            await interaction.response.send_message(embed=embed)
        
        @self.bot.tree.command(name="forget", description="Clear your conversation history")
        async def forget(interaction):
            """Clear user's session data."""
            user_id = self._get_user_id(interaction.user)
            
            # Clear caches
            if user_id in self._user_trackers:
                self._user_trackers[user_id].history.clear()
            
            await interaction.response.send_message(
                "I've forgotten our recent conversation. Let's start fresh! 🔄",
                ephemeral=True
            )
        
        @self.bot.tree.command(name="help", description="Show Lilith help")
        async def help_command(interaction):
            """Show help information."""
            embed = discord.Embed(
                title="🌙 Lilith Help",
                description="I'm Lilith, a learning conversational AI!",
                color=discord.Color.purple()
            )
            
            embed.add_field(
                name="Talking to Me",
                value=(
                    "• **In DMs**: Just send a message\n"
                    "• **In servers**: @mention me\n"
                    "• I learn from our conversations!"
                ),
                inline=False
            )
            
            embed.add_field(
                name="Commands",
                value=(
                    "`/setname` - Set your preferred name\n"
                    "`/whoami` - See what I know about you\n"
                    "`/stats` - View statistics\n"
                    "`/forget` - Clear conversation history\n"
                    "`/help` - Show this help"
                ),
                inline=False
            )
            
            embed.add_field(
                name="Feedback",
                value=(
                    "React to my messages to give feedback:\n"
                    "👍 = Good response (I'll learn from it)\n"
                    "👎 = Bad response (I'll learn to avoid it)"
                ),
                inline=False
            )
            
            embed.add_field(
                name="Teaching Me",
                value=(
                    "• Tell me facts and I'll remember them\n"
                    "• Say 'My name is X' and I'll call you that\n"
                    "• Correct me when I'm wrong"
                ),
                inline=False
            )
            
            await interaction.response.send_message(embed=embed)
    
    def run(self, token: Optional[str] = None):
        """
        Run the Discord bot.
        
        Args:
            token: Discord bot token (uses DISCORD_TOKEN env var if not provided)
        """
        token = token or os.getenv("DISCORD_TOKEN")
        
        if not token:
            raise ValueError(
                "Discord token required. Set DISCORD_TOKEN environment variable "
                "or pass token to run()"
            )
        
        print("🚀 Starting Lilith Discord Bot...")
        self.bot.run(token)
    
    async def run_with_console(self, token: str = None):
        """
        Run the bot with an interactive console for monitoring and control.
        
        Console commands:
            stats   - Show session statistics
            users   - List active users
            cleanup - Force session cleanup
            quit    - Shutdown the bot gracefully
        """
        import asyncio
        
        token = token or os.getenv("DISCORD_TOKEN")
        
        if not token:
            raise ValueError(
                "Discord token required. Set DISCORD_TOKEN environment variable "
                "or pass token to run()"
            )
        
        print("🚀 Starting Lilith Discord Bot with console...")
        print("   Type 'help' for console commands")
        print("")
        
        # Start the bot in background
        async with self.bot:
            # Start bot connection
            bot_task = asyncio.create_task(self.bot.start(token))
            
            # Wait for bot to be ready
            await self.bot.wait_until_ready()
            
            # Run console input loop
            console_task = asyncio.create_task(self._console_loop())
            
            try:
                # Wait for either task to complete
                done, pending = await asyncio.wait(
                    [bot_task, console_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                        
            except asyncio.CancelledError:
                pass
    
    async def _console_loop(self):
        """Interactive console loop running alongside the bot."""
        import asyncio
        import sys
        
        def get_input():
            """Get input in a thread-safe way."""
            try:
                return input("\n[lilith] > ").strip().lower()
            except EOFError:
                return "quit"
        
        while True:
            try:
                # Run input in executor to not block event loop
                loop = asyncio.get_event_loop()
                cmd = await loop.run_in_executor(None, get_input)
                
                if not cmd:
                    continue
                
                if cmd in ('quit', 'exit', 'q'):
                    print("👋 Shutting down...")
                    await self.bot.close()
                    break
                
                elif cmd == 'help':
                    print("""
Console Commands:
  stats     - Show session statistics
  users     - List active user sessions  
  cleanup   - Force cleanup of inactive sessions
  clear     - Clear old user data (respects retention period)
  ping      - Check if bot is responsive
  servers   - List connected servers
  quit      - Shutdown the bot gracefully
""")
                
                elif cmd == 'stats':
                    stats = self.get_session_stats()
                    print(f"""
Session Statistics:
  Active sessions: {stats['active_sessions']}
  Oldest session: {stats['oldest_session_minutes']:.1f} minutes
  Newest session: {stats['newest_session_minutes']:.1f} minutes
  Session timeout: {stats['session_timeout_minutes']} minutes
  User retention: {stats['user_retention_days']} days
""")
                
                elif cmd == 'users':
                    if not self._user_composers:
                        print("  No active user sessions")
                    else:
                        print(f"\nActive Sessions ({len(self._user_composers)}):")
                        for user_id in self._user_composers.keys():
                            last_active = self._user_last_active.get(user_id)
                            if last_active:
                                from datetime import datetime
                                age = (datetime.now() - last_active).total_seconds() / 60
                                print(f"  • {user_id} (active {age:.1f} min ago)")
                            else:
                                print(f"  • {user_id}")
                
                elif cmd == 'cleanup':
                    freed = self._cleanup_inactive_sessions()
                    print(f"  🧹 Freed {freed} inactive sessions")
                
                elif cmd == 'clear':
                    deleted = self._cleanup_old_user_data()
                    if deleted:
                        print(f"  🗑️ Deleted {len(deleted)} old user records: {deleted}")
                    else:
                        print("  No old user data to clean up")
                
                elif cmd == 'ping':
                    latency = self.bot.latency * 1000
                    print(f"  🏓 Pong! Latency: {latency:.0f}ms")
                
                elif cmd == 'servers':
                    if not self.bot.guilds:
                        print("  Not connected to any servers")
                    else:
                        print(f"\nConnected Servers ({len(self.bot.guilds)}):")
                        for guild in self.bot.guilds:
                            print(f"  • {guild.name} ({guild.member_count} members)")
                
                else:
                    print(f"  Unknown command: {cmd}")
                    print("  Type 'help' for available commands")
                    
            except Exception as e:
                print(f"  Console error: {e}")


def main():
    """Main entry point."""
    import argparse
    
    if not DISCORD_AVAILABLE:
        print("Error: discord.py is required")
        print("Install with: pip install discord.py")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='Lilith Discord Bot')
    parser.add_argument('--console', '-c', action='store_true',
                        help='Run with interactive console')
    parser.add_argument('--timeout', '-t', type=int, default=30,
                        help='Session timeout in minutes (default: 30)')
    parser.add_argument('--retention', '-r', type=int, default=7,
                        help='User data retention in days (default: 7)')
    args = parser.parse_args()
    
    bot = LilithDiscordBot(
        session_timeout_minutes=args.timeout,
        user_retention_days=args.retention
    )
    
    if args.console:
        import asyncio
        asyncio.run(bot.run_with_console())
    else:
        bot.run()


if __name__ == "__main__":
    main()
