
class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict() 
            else:
                result[key] = value
        return result