
class Color:
    # From https://en.wikipedia.org/wiki/Web_colors
    rgb = {
        'blueish' :   ['#0000FF',
                       '#9933FF',
                       '#0066FF'],
        'greenish' :  ['#00FF00',
                       '#66CC00',
                       '#009900'],
        'redish' :    ['#FF0000',
                       '#993300',
                       '#FF6600'],
        'yellowish' : ['#FFFF00',
                       '#CCCC00',
                       '#999900']
    }
    mono = {
        'grayish' :   ['#000000'
                      '#808080'
                      '#FFFFFF']
    }
    #blue = blueish[0]
    #red = redish[0]
    #green = greenish[0]
    #yellow = yellowish[0]

    @classmethod
    def get_colors(this,N,mode='sparse'):
        """
        mode = sparse
        """
        mode = mode if mode is 'sparse' else 'sparse'
        from collections import OrderedDict
        if mode is 'sparse':
            groups = this.rgb.keys()
            groups.sort()
            color_groups = OrderedDict()
            for g in groups:
                color_groups[g] = this.rgb.get(g)[:]

        #TODO: make this selection better, probably objectfying each color group
        colors = []
        while len(colors)<N:
            for _g in color_groups.keys():
                try:
                    colors.append(color_groups[_g].pop(0))
                except IndexError as e:
                    raise IndexError("You're asking me {0} colors, more than I have to offer".format(N))
                if len(colors) == N:
                       break
        return colors

Colors = Color
