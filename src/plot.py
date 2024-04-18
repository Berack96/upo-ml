import matplotlib.pyplot as plt
from typing_extensions import Self

class Plot:
    def __init__(self, title:str, labelx:str, labely:str) -> None:
        plt.title(title)
        plt.xlabel(labelx)
        plt.ylabel(labely)
        plt.ion()
        plt.show(block=False)

        self.data = dict()

    def wait(self) -> Self:
        plt.ioff()
        plt.show()
        return self

    def scatter(self, label:str, datax:list[float], datay:list[float], color:str) -> Self:
        plt.scatter(datax, datay, color=color, label=label)
        return self

    def line(self, label:str, color:str, data:list[float]=[], max_length:int=100) -> Self:
        line, = plt.plot(data if len(data) > 0 else [0], label=label, color=color)
        x = [] if len(data) == 0 else [*range(len(data))]

        self.data[label] = (line, data, x, max_length)
        plt.legend()
        return self

    def update(self, label:str, newdata:float) -> Self:
        line, datay, datax, max = self.data[label]

        x = 0 if len(datax) == 0 else datax[-1]
        datax.append(x+1)
        datay.append(newdata)

        remove = len(datax) - max
        if remove > 0:
            del datax[:remove]
            del datay[:remove]

        line.set_data((datax, datay))
        return self

    def update_limits(self) -> Self:
        if not bool(plt.get_fignums()): raise Exception("plot closed!")
        limy_top = 0.1
        limx_top, limx_bot = (0, 100000000000000000)

        for val in self.data:
            _, datay, datax, _ = self.data[val]
            limy_top = max(max(datay), limy_top)
            limx_top = max(max(datax), limx_top)
            limx_bot = min(min(datax), limx_bot)
        if limx_top == limx_bot: limx_top += 1

        plt.xlim(limx_bot, limx_top)
        plt.ylim(0, limy_top)
        plt.draw()
        plt.pause(0.0000000001)
        return self

