from typing import Dict, List, Any, Optional, Protocol, Union
import logging
from dataclasses import dataclass

from .io_port import IOPort
from .cognitive_stage import CognitiveStage
from .mcp_router import EndpointType

logger = logging.getLogger(__name__)

@dataclass
class SomaticConfig:
    sense_ports: List[str] # IDs of ports treated as Input (Senses)
    limb_ports: List[str]  # IDs of ports treated as Output (Limbs)

class SomaticLayer:
    """
    The 'Body' of Lilith.
    Connects the 'Brain' (CognitiveStage) to the 'World' (MCP IOPorts).
    
    Responsibilities:
    1. Proprioception: Monitoring connected ports.
    2. Afferent Flow: Routing Port Input -> Cognitive Stage (Senses).
    3. Efferent Flow: Routing Cognitive Output -> Port Output (Limbs).
    """
    
    def __init__(
        self, 
        brain: CognitiveStage,
        ports: Dict[str, IOPort],
        config: SomaticConfig
    ):
        self.brain = brain
        self.ports = ports
        self.config = config
        
    def tick(self):
        """Single heartbeat of the body."""
        self._process_senses()
        self._process_actions()
        
    def _process_senses(self):
        """Gather input from all 'Sense' ports and feed to Brain."""
        for pid in self.config.sense_ports:
            port = self.ports.get(pid)
            if not port: continue
            
            # Drain port
            # Assuming port.receive() is non-blocking generator
            # Pass empty meta as required by IOPort protocol
            for msg in port.receive(meta={}):
                logger.debug(f"[Somatic] Check sense {pid}: {msg}")
                # For raw messages (dicts), pass them directly instead of str()
                # allow complex structures to reach the brain
                payload = msg if isinstance(msg, (dict, list)) else str(msg)
                
                self.brain.learn(payload, ctx={"source_sense": pid})

    def _process_actions(self):
        """Check if Brain has volition to act, and route to 'Limb' ports."""
        # Brain needs a way to signal action validation.
        # Currently CognitiveStage writes to self.last_interaction['response']
        
        last_out = self.brain.last_interaction
        if not last_out: return
        
        response_text = last_out.get("response")
        # In tests we set this complete object as the output intention
        # Use the whole object if it seems structured, or falling back
        payload = last_out if (isinstance(last_out, dict) and "action" in last_out) else response_text
        
        if not payload: return

        # Simple broadcast to all limbs (Speech, Console, VSCode)
        # Real impl would use Attention/Selection
        for pid in self.config.limb_ports:
            port = self.ports.get(pid)
            if not port: continue
            
            # Send message
            port.send(payload, meta={"source": "cognitive_stage"})
            
        # Clear impulse (Refractory period)
        self.brain.last_interaction = None
