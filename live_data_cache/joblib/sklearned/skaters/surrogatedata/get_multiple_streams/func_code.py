# first line: 13
def get_multiple_streams(sub_sample=2, include_strs=None)->List[List[float]]:
    """  """
    if include_strs is None:
        include_strs = ['hospital','electricity','airport','volume','emoji','three_body','helicopter','noaa']
    mr = MicroReader()
    streams = mr.get_stream_names()
    acceptable = [s for s in streams if any( [incl in s for incl in include_strs]) and not '~' in s]
    ys = list()
    for nm in acceptable:
        try:
            lagged_values, lagged_times = mr.get_lagged_values_and_times(name=nm, count=2000)
            y, t = list(reversed(lagged_values)), list(reversed(lagged_times))
            if 'hospital' in nm:
                y_sub = y[::sub_sample]
            else:
                y_sub = y
            y_scale = np.mean([ abs(yi) for yi in y_sub[:100]])+1
            if len(y_sub)>=750:
                y_scaled = [ yi/y_scale for yi in y_sub]
                ys.append(y_scaled)
                print(nm)
            else:
                print(nm+' too short')
        except:
            print(nm+' exception')
    print(len(ys))
    return ys
