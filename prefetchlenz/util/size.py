class Size:
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB

    def __init__(self, bytes: int):
        self.bytes = bytes

    def __add__(self, other):
        return Size(self.bytes + int(other))

    def __sub__(self, other):
        return Size(self.bytes - int(other))

    def __int__(self):
        return self.bytes

    def __repr__(self):
        if self.bytes >= Size.GB:
            return f"{self.bytes / Size.GB:.2f} GB"
        elif self.bytes >= Size.MB:
            return f"{self.bytes / Size.MB:.2f} MB"
        elif self.bytes >= Size.KB:
            return f"{self.bytes / Size.KB:.2f} KB"
        else:
            return f"{self.bytes} B"

    @classmethod
    def from_kb(cls, value: float):
        return cls(int(value * cls.KB))

    @classmethod
    def from_mb(cls, value: float):
        return cls(int(value * cls.MB))

    @classmethod
    def from_gb(cls, value: float):
        return cls(int(value * cls.GB))
