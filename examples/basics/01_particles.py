from tolvera import Tolvera

def main(**kwargs):
    tv = Tolvera(**kwargs)

    @tv.render
    def _():
        pass
