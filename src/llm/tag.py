
class Tag:
    def __init__(self, model, method, task, temperature=0.7,**kwargs):
        self.model = model
        self.temperature = temperature
        self.method = method
        self.task = task
        self.configs = kwargs

    def __str__(self):
        return f"model={self.model}, method={self.method}, task={self.task}, temperature={self.temperature}, configs={self.configs}"
