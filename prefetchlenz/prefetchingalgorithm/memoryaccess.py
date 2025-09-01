from dataclasses import dataclass


@dataclass
class MemoryAccess:
    cpu: int
    address: int
    pc: int
