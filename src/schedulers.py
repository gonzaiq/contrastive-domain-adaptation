from typing import Union, List

class StepVariableScheduler:
    def __init__(self, init_value: Union[float, int, bool], final_value: Union[float, int, bool], step_size: int):

        self.value = init_value
        self.final_value = final_value
        self.step_size = step_size
        self.epoch = 0

    def step(self):
        """Update the variable based on the step schedule."""
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.value = self.final_value

    def get_value(self):
        """Returns the current value of the variable."""
        return self.value

class MultiStepVariableScheduler:
    def __init__(self, init_value: Union[float, int, bool], values: List[Union[float, int, bool]], steps: List[int]):
        """
        Args:
            init_value (float): Initial value of the variable.
            values (list of float): Sequence of values to assign.
            steps (list of int): Sequence of steps at which to change the value.
                                 Must be the same length as `values`.
        """
        assert len(values) == len(steps), "values and steps must have the same length"
        
        self.value = init_value
        self.values = values
        self.steps = steps
        self.epoch = 0

    def step(self):
        """Update the variable based on the step schedule."""
        self.epoch += 1
        print(self.epoch, self.steps, self.values, self.epoch in self.steps)
        if self.epoch in self.steps:
            print("in!")
            index = self.steps.index(self.epoch)
            self.value = self.values[index]
            print(index, self.value)

    def get_value(self):
        """Returns the current value of the variable."""
        return self.value

class LinearVariableScheduler:
    def __init__(self, init_value: Union[int, float], final_value: Union[int, float], total_epochs: int):
        """
        Args:
            init_value (float): Initial value of the variable.
            final_value (float): Final value of the variable after decay.
            total_epochs (int): Number of epochs over which to linearly decay the value.
        """
        self.init_value = init_value
        self.final_value = final_value
        self.total_epochs = total_epochs
        self.value = init_value
        self.epoch = 0

    def step(self):
        """Update the variable by linearly decreasing it."""
        if self.epoch < self.total_epochs:
            self.value = self.init_value - (self.init_value - self.final_value) * (self.epoch / self.total_epochs)
            self.epoch += 1
        else:
            self.value = self.final_value

    def get_value(self):
        """Returns the current value of the variable."""
        return self.value