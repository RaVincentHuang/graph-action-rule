
class Tag:
    def __init__(self, model, method, task, temperature=0.7,**kwargs):
        self.model = model
        self.temperature = temperature
        self.method = method
        self.task = task
        self.configs = kwargs

    def __str__(self):
        return f"model={self.model}, method={self.method}, task={self.task}, temperature={self.temperature}, configs={self.configs}"

    @staticmethod
    def from_str(s):
        parts = s.split(',')
        model = parts[0].split('=')[1]
        method = parts[1].split('=')[1]
        task = parts[2].split('=')[1]
        temperature = float(parts[3].split('=')[1])
        configs = {}
        for part in parts[4:]:
            key, value = part.split('=')
            configs[key] = value
        return Tag(model, method, task, temperature, **configs)