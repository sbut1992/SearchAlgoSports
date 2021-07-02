import pstats
from pstats import SortKey
p = pstats.Stats('profiling')
p.sort_stats(SortKey.CUMULATIVE).print_stats(100)