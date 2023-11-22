'''
TODO: 'sketchbook' run/export scripts?
'''

import fire

from tolvera import Tolvera

def help():
    print('')

def main(**kwargs):
    tv = Tolvera(**kwargs)

    @tv.render
    def _():
        tv.px.diffuse(0.99)
        tv.p()
        tv.v.flock(tv.p)
        tv.px.particles(tv.p, tv.s, 'circle')
        return tv.px

if __name__=='__main__':
    fire.Fire(main)
