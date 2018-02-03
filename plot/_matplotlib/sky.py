__FIGSIZE=(8,6)

def __guarantee_angle(coord,unit):
    from astropy import coordinates
    if not isinstance(coord, coordinates.Angle):
        coord = coordinates.Angle(coord, unit=unit)
    else:
        coord = coord.to(unit)
    return coord


def scatter_sky(ra, dec, projection='aitoff', all_sky=True,
                marker='o', color='b', label='sky-scatter plot'):
    '''
    Input:
     - ra
     - dec
     - unit: string
        Options are: degree, sexagesimal
     - projection: string
        Options are: hammer, mollweide, lambert, aitoff
     - crop: bool
        If 'True', crop plot to content

    Output:
     - ~matplotlib.pyplot.figure
    '''
    assert hasattr(ra,'unit')
    assert hasattr(dec,'unit')

    unit = 'degree'

    from astropy import units
    _u = units.Unit(unit)

    ra = __guarantee_angle(ra, unit)
    ra = ra.wrap_at(180*units.degree)
    dec = __guarantee_angle(dec, unit)

    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=__FIGSIZE)
    if all_sky is True:
        ax = fig.add_subplot(111, projection=projection)
        if _u != units.degree:
            ax.set_xticklabels(['14h','16h','18h','20h','22h','0h','2h','4h','6h','8h','10h'])
        ax.set_title(label)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.grid(True)
        ax.scatter(ra.radian, dec.radian, marker=marker, color=color)
    else:
        from . import scatter
        fig = scatter.scatter(ra.degree,dec.degree)

    return fig
