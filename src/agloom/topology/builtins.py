"""Registry of Agloom's eight built-in execution topologies."""

from agloom.models import TopologyName
from agloom.topology.base import Topology
from agloom.topology.blackboard import BlackboardTopology
from agloom.topology.hybrid import HybridTopology
from agloom.topology.pipeline import PipelineTopology
from agloom.topology.planner import PlannerTopology
from agloom.topology.react import ReactTopology
from agloom.topology.reflection import ReflectionTopology
from agloom.topology.supervisor import SupervisorTopology
from agloom.topology.swarm import SwarmTopology

BUILTIN_TOPOLOGIES: dict[TopologyName, Topology] = {
    TopologyName.REACT: ReactTopology(),
    TopologyName.SUPERVISOR: SupervisorTopology(),
    TopologyName.PIPELINE: PipelineTopology(),
    TopologyName.PLANNER: PlannerTopology(),
    TopologyName.REFLECTION: ReflectionTopology(),
    TopologyName.SWARM: SwarmTopology(),
    TopologyName.BLACKBOARD: BlackboardTopology(),
    TopologyName.HYBRID: HybridTopology(),
}
