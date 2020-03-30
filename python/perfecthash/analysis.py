#===============================================================================
# Imports
#===============================================================================

#===============================================================================
# Globals
#===============================================================================

HASH_FUNCTIONS = (
    'Jenkins',
    'RotateMultiplyXorRotate',
    'RotateMultiplyXorRotate2',
    'ShiftMultiplyXorShift',
    'ShiftMultiplyXorShift2',
    'MultiplyRotateR',
    'MultiplyRotateLR',
    'MultiplyShiftR',
)

BEST_COVERAGE_TYPES = (
    'HighestNumberOfEmptyCacheLines',
    'HighestMaxGraphTraversalDepth',
    'HighestTotalGraphTraversals',
    'HighestMaxAssignedPerCacheLineCount',
    'LowestNumberOfEmptyCacheLines',
    'LowestMaxGraphTraversalDepth',
    'LowestTotalGraphTraversals',
    'LowestMaxAssignedPerCacheLineCount',
)

KEYS_SLIM_1 = [
    'KeysName',
    'HashFunction',
    'BestCoverageType',
    'KeysToEdgesRatio',
    'SolutionsFoundRatio',
    'NumberOfKeys',
    'NumberOfEdges',
    'Attempts',
    'NumberOfSolutionsFound',
    'NewBestGraphCount',
    'AttemptThatFoundBestGraph',
    'ComputerName',
    'ContextTimestamp',
    'TableTimestamp',
    'Version',
    'VerifyMicroseconds',
    'DeltaHashMinimumCycles',
]

KEYS_BEST_COVERAGE_1 = [
    'BestCoverageSlope',
    'BestCoverageIntercept',
    'BestCoverageRValue',
    'BestCoveragePValue',
    'BestCoverageStdErr',
    'BestCoverageSlope',
    'BestCoverageValue',
    'BestCoverageEqualCount',
]

#===============================================================================
# Helper Functions
#===============================================================================

def df_from_csv(path):
    d = dirname(path)
    (sys, group) = d.split('/')
    df = pd.read_csv(path)
    df['System'] = sys
    df['Group'] = group
    return df

def update_df_old(df):
    from tqdm import tqdm
    import numpy as np
    from scipy.stats import linregress
    df['BestCoverageValue'] = np.int(0)
    df['BestCoverageEqualCount'] = np.int(0)
    df['BestCoverageSlope'] = np.float(0)
    df['BestCoverageIntercept'] = np.float(0)
    df['BestCoverageRValue'] = np.float(0)
    df['BestCoverageR2'] = np.float(0)
    df['BestCoveragePValue'] = np.float(0)
    df['BestCoverageStdErr'] = np.float(0)
    #df['BestCoverageCovariance'] = np.float(0)
    df['BestCoveragePositiveSlopeNumber'] = np.int(0)
    df['BestCoveragePositiveSlopeAttempt'] = np.int(0)
    df['KeysToEdgesRatio'] = df.NumberOfKeys / df.NumberOfEdges
    df['SolutionsFoundRatio'] = df.NumberOfSolutionsFound / df.Attempts
    x = np.array(list(range(0, 17)))[:, np.newaxis]
    x_flat = np.array(list(range(0, 17)))
    count_keys = [
        f'CountOfCacheLinesWithNumberOfAssigned_{n}'
            for n in range(0, 17)
    ]
    for (i, row) in tqdm(df.iterrows(), total=len(df)):
        best_count = row['NewBestGraphCount']
        if best_count == 0:
            # No solution was found.
            continue
        if best_count > 32:
            # We only capture up to 32 best graph attempts, in practice, it's
            # rare to get anything higher than 32 here.
            best_count = 32

        best_value = row[f'BestGraph{best_count}_Value']
        best_equal_count = row[f'BestGraph{best_count}_EqualCount']
        df.at[i, 'BestCoverageValue'] = np.int(best_value)
        df.at[i, 'BestCoverageEqualCount'] = np.int(best_equal_count)

        y = df.iloc[i][count_keys].values.astype(np.int)

        r = linregress(x_flat, y)

        df.at[i, 'BestCoverageSlope'] = r.slope
        df.at[i, 'BestCoverageIntercept'] = r.intercept
        df.at[i, 'BestCoverageRValue'] = r.rvalue
        df.at[i, 'BestCoverageR2'] = r.rvalue**2
        df.at[i, 'BestCoveragePValue'] = r.pvalue
        df.at[i, 'BestCoverageStdErr'] = r.stderr
        #df.at[i, 'BestCoverageCovariance'] = np.cov(x_flat, y)

        version = int(row['Version'][1:].split('.')[1])
        if version < 38:
            continue

        positive_slope = None
        attempt = None

        for j in range(1, best_count+1):
            best_keys = [
                f'BestGraph{j}_CountOfCacheLinesWithNumberOfAssigned_{n}'
                    for n in range(0, 17)
            ]

            y = df.iloc[i][best_keys].values.astype(np.int)
            r = linregress(x_flat, y)
            r2 = r.rvalue**2

            df.at[i, f'BestGraph{j}_Slope'] = r.slope
            df.at[i, f'BestGraph{j}_Intercept'] = r.intercept
            df.at[i, f'BestGraph{j}_RValue'] = r.rvalue
            df.at[i, f'BestGraph{j}_R2'] = r2
            df.at[i, f'BestGraph{j}_PValue'] = r.pvalue
            df.at[i, f'BestGraph{j}_StdErr'] = r.stderr
            #df.at[i, f'BestGraph{j}_Covariance'] = np.cov(x_flat, y)

            if j == 1 or positive_slope is not None:
                continue

            if r.slope > 0 and r2 > 70.0:
                positive_slope = j
                attempt = df.iloc[i]['BestGraph{j}_Attempt'].values[0]

        if positive_slope is not None:
            df[i, 'BestCoveragePositiveSlopeNumber'] = j
            df[i, 'BestCoveragePositiveSlopeAttempt'] = attempt
        else:
            df[i, 'BestCoveragePositiveSlopeNumber'] = 0
            df[i, 'BestCoveragePositiveSlopeAttempt'] = 0

def update_df(df, source_csv_file):
    from tqdm import tqdm
    import numpy as np
    from scipy.stats import linregress
    df['SourceCsvFile'] = source_csv_file
    df['BestCoverageValue'] = np.int(0)
    df['BestCoverageEqualCount'] = np.int(0)
    df['BestCoverageSlope'] = np.float(0)
    df['BestCoverageIntercept'] = np.float(0)
    df['BestCoverageRValue'] = np.float(0)
    df['BestCoverageR2'] = np.float(0)
    df['BestCoveragePValue'] = np.float(0)
    df['BestCoverageStdErr'] = np.float(0)
    #df['BestCoverageCovariance'] = np.float(0)
    df['BestCoveragePositiveSlopeNumber'] = np.int(0)
    df['BestCoveragePositiveSlopeAttempt'] = np.int(0)
    df['KeysToEdgesRatio'] = df.NumberOfKeys / df.NumberOfEdges
    df['KeysToVerticesRatio'] = df.NumberOfKeys / df.NumberOfVertices
    df['SolutionsFoundRatio'] = df.NumberOfSolutionsFound / df.Attempts
    x = np.array(list(range(0, 17)))[:, np.newaxis]
    x_flat = np.array(list(range(0, 17)))
    count_keys = [
        f'CountOfCacheLinesWithNumberOfAssigned_{n}'
            for n in range(0, 17)
    ]
    if 'InitialNumberOfTableResizes' not in df.columns:
        df['InitialNumberOfTableResizes'] = np.int(0)
    for (i, row) in tqdm(df.iterrows(), total=len(df)):
        best_count = row['NewBestGraphCount']
        if best_count == 0:
            # No solution was found.
            continue
        if best_count > 32:
            # We only capture up to 32 best graph attempts, in practice, it's
            # rare to get anything higher than 32 here.
            best_count = 32

        version = int(row['Version'][1:].split('.')[1])
        if version < 38:
            continue

        best_value = row[f'BestGraph{best_count}_Value']
        best_equal_count = row[f'BestGraph{best_count}_EqualCount']
        df.at[i, 'BestCoverageValue'] = np.int(best_value)
        df.at[i, 'BestCoverageEqualCount'] = np.int(best_equal_count)

        y = df.iloc[i][count_keys].values.astype(np.int)

        r = linregress(x_flat, y)

        df.at[i, 'BestCoverageSlope'] = r.slope
        df.at[i, 'BestCoverageIntercept'] = r.intercept
        df.at[i, 'BestCoverageRValue'] = r.rvalue
        df.at[i, 'BestCoverageR2'] = r.rvalue**2
        df.at[i, 'BestCoveragePValue'] = r.pvalue
        df.at[i, 'BestCoverageStdErr'] = r.stderr
        #df.at[i, 'BestCoverageCovariance'] = np.cov(x_flat, y)

def linregress_hash_func_by_number_of_edges(df):
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from itertools import product
    from scipy.stats import linregress

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    num_edges = (
        df.NumberOfEdges
            .value_counts()
            .sort_index()
            .index
            .values
    )

    keys_subset = [
        'HashFunction',
        'NumberOfEdges',
        'KeysToEdgesRatio',
        'KeysToVerticesRatio',
        'SolutionsFoundRatio',
    ]

    dfa = df[keys_subset]

    targets = [ (hf, ne) for (hf, ne) in product(hash_funcs, num_edges) ]

    results = []

    np.seterr('raise')

    for (hash_func, num_edges) in tqdm(targets):

        query = (
            f'HashFunction == "{hash_func}" and '
            f'NumberOfEdges == {num_edges}'
        )
        df = dfa.query(query)

        # Skip empties.
        if df.empty:
            continue

        keys_to_edges_ratio = df.KeysToEdgesRatio.values
        solutions_found_ratio = df.SolutionsFoundRatio.values

        count = len(keys_to_edges_ratio)

        # linregress won't work if all the x values are the same.
        if len(np.unique(keys_to_edges_ratio)) == 1:
            continue

        lr = linregress(keys_to_edges_ratio, solutions_found_ratio)

        #cycles = df.DeltaHashMinimumCycles.describe(percentiles=[0.95])['95%']
        #cycles = int(cycles)

        label = (
            f'{hash_func} ({count}): '
            f'y={lr.slope:.2f}x + {lr.intercept:.02f} '
            f'[r: {lr.rvalue:.3f}, p: {lr.pvalue:.3f}, '
            f'stderr: {lr.stderr:.3f}]'
        )

        result = (
            hash_func,
            num_edges,
            lr.slope,
            lr.intercept,
            lr.rvalue,
            lr.pvalue,
            lr.stderr,
            count,
            label,
            keys_to_edges_ratio,
            solutions_found_ratio,
        )

        results.append(result)

    columns = [
        'HashFunction',
        'NumberOfEdges',
        'Slope',
        'Intercept',
        'RValue',
        'PValue',
        'StdErr',
        'Count',
        'Label',
        'KeysToEdgesRatio',
        'SolutionsFoundRatio',
    ]

    return pd.DataFrame(results, columns=columns)

def linregress_hash_func_by_number_of_vertices(df):
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from itertools import product
    from scipy.stats import linregress

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    num_vertices = (
        df.NumberOfVertices
            .value_counts()
            .sort_index()
            .index
            .values
    )

    keys_subset = [
        'HashFunction',
        'NumberOfVertices',
        'KeysToVerticesRatio',
        'SolutionsFoundRatio',
    ]

    dfa = df[keys_subset]

    targets = [ (hf, ne) for (hf, ne) in product(hash_funcs, num_vertices) ]

    results = []

    np.seterr('raise')

    for (hash_func, num_vertices) in tqdm(targets):

        query = (
            f'HashFunction == "{hash_func}" and '
            f'NumberOfVertices == {num_vertices}'
        )
        df = dfa.query(query)

        # Skip empties.
        if df.empty:
            continue

        keys_to_vertices_ratio = df.KeysToVerticesRatio.values
        solutions_found_ratio = df.SolutionsFoundRatio.values

        count = len(keys_to_vertices_ratio)

        # linregress won't work if all the x values are the same.
        if len(np.unique(keys_to_vertices_ratio)) == 1:
            continue

        lr = linregress(keys_to_vertices_ratio, solutions_found_ratio)

        label = (
            f'{hash_func} ({count}): '
            f'y={lr.slope:.2f}x + {lr.intercept:.02f} '
            f'[r: {lr.rvalue:.3f}, p: {lr.pvalue:.3f}, '
            f'stderr: {lr.stderr:.3f}]'
        )

        result = (
            hash_func,
            num_vertices,
            lr.slope,
            lr.intercept,
            lr.rvalue,
            lr.pvalue,
            lr.stderr,
            count,
            label,
            keys_to_vertices_ratio,
            solutions_found_ratio,
        )

        results.append(result)

    columns = [
        'HashFunction',
        'NumberOfVertices',
        'Slope',
        'Intercept',
        'RValue',
        'PValue',
        'StdErr',
        'Count',
        'Label',
        'KeysToVerticesRatio',
        'SolutionsFoundRatio',
    ]

    return pd.DataFrame(results, columns=columns)

def find_non_identical_solving_ratios_per_hash_func_and_keys(df):

    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from itertools import product
    from scipy.stats import linregress

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    keys_names = (
        df.KeysName
            .value_counts()
            .sort_index()
            .index
            .values
    )

    keys_subset = [
        'HashFunction',
        'KeysName',
        'KeysToEdgesRatio',
        'SolutionsFoundRatio',
    ]

    dfa = df[keys_subset]

    targets = [ (hf, kn) for (hf, kn) in product(hash_funcs, keys_names) ]

    results = []

    np.seterr('raise')

    #import ipdb
    #ipdb.set_trace()

    for (hash_func, keys_name) in tqdm(targets):

        query = (
            f'HashFunction == "{hash_func}" and '
            f'KeysName == "{keys_name}"'
        )
        df = dfa.query(query)

        # Value count of solutions found... if not more than 1, continue,
        # otherwise... print.
        vc = df.SolutionsFoundRatio.value_counts()
        if len(vc) <= 1:
            continue

        results.append((hash_func, keys_name, vc))

    return results

def extract_mean_solving_data(df):

    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from itertools import product
    from scipy.stats import linregress

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    keys_names = (
        df.KeysName
            .value_counts()
            .sort_index()
            .index
            .values
    )

    num_edges = (
        df.NumberOfEdges
            .value_counts()
            .sort_index()
            .index
            .values
    )

    num_resizes = (
        df.NumberOfTableResizeEvents
            .value_counts()
            .sort_index()
            .index
            .values
    )

    keys_subset = [
        'KeysName',
        'HashFunction',
        'NumberOfTableResizeEvents',
        'NumberOfKeys',
        'NumberOfEdges',
        'NumberOfVertices',
        'KeysToEdgesRatio',
        'KeysToVerticesRatio',
        'SolutionsFoundRatio',
    ]

    dfa = df[keys_subset]

    targets = [
        (hf, kn, ne, nr) for (hf, kn, ne, nr) in (
            product(hash_funcs, keys_names, num_edges, num_resizes)
        )
    ]

    results = []

    np.seterr('raise')

    for (hash_func, keys_name, num_edges, num_resizes) in tqdm(targets):

        query = (
            f'HashFunction == "{hash_func}" and '
            f'KeysName == "{keys_name}" and '
            f'NumberOfEdges == {num_edges} and '
            f'NumberOfTableResizeEvents == {num_resizes}'
        )
        df = dfa.query(query)

        # Skip empties.
        if df.empty:
            continue

        result = [
            keys_name,
            hash_func,
            df.NumberOfTableResizeEvents.values[0],
            df.NumberOfKeys.values[0],
            df.NumberOfEdges.values[0],
            df.NumberOfVertices.values[0],
            df.KeysToEdgesRatio.values[0],
            df.KeysToVerticesRatio.values[0],
        ]

        result += list(df.SolutionsFoundRatio.describe().values)

        results.append(result)

    suffixes = [
        '_Count',
        '', # Mean
        '_StdDev',
        '_Min',
        '_25%',
        '_50%',
        '_75%',
        '_Max',
    ]

    columns = [
        'KeysName',
        'HashFunction',
        'NumberOfTableResizeEvents',
        'NumberOfKeys',
        'NumberOfEdges',
        'NumberOfVertices',
        'KeysToEdgesRatio',
        'KeysToVerticesRatio',
    ]

    columns += [
        f'SolutionsFoundRatio{suffix}' for suffix in suffixes
    ]

    new_df = pd.DataFrame(results, columns=columns)
    return new_df

def perf_details_by_hash_func(df):
    import pandas as pd
    from tqdm import tqdm
    from itertools import product
    from scipy.stats import linregress

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    num_edges = (
        df.NumberOfEdges
            .value_counts()
            .sort_index()
            .index
            .values
    )

    keys_subset = [
        'HashFunction',
        'NumberOfKeys',
        'NumberOfEdges',
        'VerifyMicroseconds',
        'DeltaHashMinimumCycles',
    ]

    dfa = df[keys_subset]

    targets = [ (hf, ne) for (hf, ne) in product(hash_funcs, num_edges) ]

    results = []

    for (hash_func, num_edges) in tqdm(targets):

        query = (
            f'HashFunction == "{hash_func}" and '
            f'NumberOfEdges == {num_edges}'
        )
        df = dfa.query(query)

        x = df.NumberOfKeys.values
        y = df.VerifyMicroseconds.values

        lr = linregress(x, y)

        result = (
            hash_func,
            num_edges,
            lr.slope,
            lr.intercept,
            lr.rvalue,
            lr.pvalue,
            lr.stderr,
            len(x),
            x,
            y,
            y / x,
        )

        results.append(result)

    columns = [
        'HashFunction',
        'NumberOfEdges',
        'Slope',
        'Intercept',
        'RValue',
        'PValue',
        'StdErr',
        'Count',
        'NumberOfKeys',
        'VerifyMicroseconds',
        'VerifyMicrosecondsPerKey',
    ]

    return pd.DataFrame(results, columns=columns)

#===============================================================================
# Format Conversion
#===============================================================================

def df_to_parquet(df, path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, compression='snappy')
    return table

def df_to_feather(df, path):
    import pyarrow.feather
    return pyarrow.feather.write_feather(df, path)

def df_from_parquet(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pandas()

def get_yyyy_mm_dd_subdirs(dirname):
    import re
    import glob
    from os.path import isdir
    pattern = re.compile(r'\d{4}[-/]\d{2}[-/]\d{2}')
    subdirs = [
        subdir for subdir in glob.iglob('*') if (
            isdir(subdir) and
            pattern.match(subdir)
        )
    ]
    return subdirs

def get_csv_files(directory):
    import glob
    return [
        f for f in glob.iglob(
            f'{directory}/**/PerfectHashBulkCreate*.csv',
            recursive=True
        )
    ]

def get_all_bulk_create_parquet_files(directory):
    import glob
    return [
        f for f in glob.iglob(
            f'{directory}/**/PerfectHashBulkCreate*.parquet',
            recursive=True
        ) if 'failed' not in f
    ]

def get_best_bulk_create_parquet_files(directory):
    import glob
    return [
        f for f in glob.iglob(
            f'{directory}/**/PerfectHashBulkCreateBest*.parquet',
            recursive=True
        ) if 'failed' not in f
    ]

def convert_csv_to_parquet(path, base_research_dir, out=None):
    if not out:
        out = lambda _: None

    from os.path import exists
    import pandas as pd

    df = None
    sdf = None

    parquet_path = path.replace('.csv', '.parquet')
    if 'failed' in parquet_path:
        return

    if not exists(parquet_path):
        out(f'Processing {path} -> {parquet_path}...')

        df = pd.read_csv(path)
        source_csv_file = path.replace(base_research_dir + '\\', '')
        update_df(df, source_csv_file)

        df_to_parquet(df, parquet_path)
        out(f'Wrote {parquet_path}.')

def concat_subdir_parquets(base, subdir, out=None):
    if not out:
        out = lambda _: None

    from os.path import exists
    import pandas as pd
    from tqdm import tqdm

    subdir_path = f'{base}\\{subdir}'
    results_path = f'{base}\\{subdir}\\results.parquet'
    if not exists(results_path):
        names = get_all_bulk_create_parquet_files(subdir_path)
        if not names:
            out('No .parquet files found to concatenate.')
            return
        out(f'Processing all {len(names)} file(s) in {subdir}:')
        out('\n'.join(f'    {name}' for name in names))
        dfs = [ df_from_parquet(n) for n in tqdm(names) ]
        df = pd.concat(dfs, sort=False, ignore_index=True)
        df_to_parquet(df, results_path)
        out(f'Wrote {results_path}.')
        return

    df = df_from_parquet(results_path)
    processed_source_csv_files = set(
        df.SourceCsvFile
            .value_counts()
            .index
            .values
    )

    files = get_best_bulk_create_parquet_files(subdir_path)

    found_source_csv_files = set(
        f.replace(base + '\\', '')
         .replace('.parquet', '.csv') for f in files
    )

    missing = found_source_csv_files - processed_source_csv_files

    if not missing:
        out(f'Up-to-date: {subdir_path}.')
        return

    out(f'Adding {len(missing)} new file(s) to {results_path}:')
    out('\n'.join(f'    {m}' for m in missing))

    names = [
        f'{base}\\{subpath}'.replace('.csv', '.parquet')
            for subpath in missing
    ]

    dfs = [ df, ] + [ df_from_parquet(n) for n in tqdm(names) ]

    df = pd.concat(dfs, sort=False, ignore_index=True)
    df_to_parquet(df, results_path)
    out(f'Wrote {results_path}.')
    return


def post_process_results_parquet(base, subdir, out=None):
    if not out:
        out = lambda _: None

    from os.path import exists
    import pandas as pd

    df = None
    sdf = None

    parquet_path = f'{base}\\{subdir}\\results.parquet'
    if not exists(parquet_path):
        out(f'{parquet_path} does not exist, run concat-parquet-to-results-'
            f'parquet command first.')
        return

    # Solutions found ratio.
    sfr_path = parquet_path.replace('.parquet', '-sfr.parquet')
    if not exists(sfr_path):
        out(f'Processing {parquet_path} -> {sfr_path}...')

        if df is None:
            df = df_from_parquet(parquet_path)

        sdf = extract_mean_solving_data(df)
        df_to_parquet(sdf, sfr_path)
        out(f'Wrote {sfr_path}.')

    # Linear regressions.
    lr_path = parquet_path.replace('.parquet', '-lr.parquet')
    if not exists(lr_path):
        out(f'Processing {sfr_path} -> {lr_path}...')

        if sdf is None:
            sdf = df_from_parquet(sfr_path)

        ldf = linregress_hash_func_by_number_of_vertices(sdf)
        df_to_parquet(ldf, lr_path)
        out(f'Wrote {lr_path}.')

#===============================================================================
# Plotting
#===============================================================================

def get_cache_line_coverage(df):
    count = df.NewBestGraphCount.values[0]
    keys = [
        f'BestGraph{i}_CountOfCacheLinesWithNumberOfAssigned_{n}'
            for i in range(1, count+1)
                for n in range(0, 17)
    ]
    values = df[keys].values.astype(np.int)
    values = values.reshape(count, 17)
    attempts = df[[f'BestGraph{i}_Attempt' for i in range(1, count+1)]].values[0]
    columns = [ f'{i}' for i in range(0, 17) ]
    return (keys, values, attempts, columns)

def ridgeline_plot(df):
    import matplotlib.pyplot as plt
    plt.ioff()
    keys_name = df.KeysName.values[0]
    hash_func = df.HashFunction.values[0]
    best_coverage_type = df.BestCoverageType.values[0]
    table_timestamp = df.TableTimestamp.values[0]
    (keys, values, attempts, columns) = get_cache_line_coverage(df)
    dfc = pd.DataFrame(values, columns=columns, index=attempts)
    dft = dfc.T
    dft = dft[[ c for c in reversed(sorted(dft.columns)) ]]
    x_range = list(range(0, 17))
    (figure, axes) = joypy.joyplot(
        dft,
        kind="values",
        x_range=x_range,
        labels=attempts,
        colormap=cm.rainbow,
        title=f"{keys_name} {hash_func}\n{best_coverage_type}\n{table_timestamp}",
        alpha=0.5,
        overlap=0.5,
        ylim=True,
        grid=True,
        figsize=(8, 8),
    )
    axes[-1].set_xticks(x_range)
    #plt.draw()
    #plt.cla()
    #plt.clf()
    return (figure, axes)


def ridgeline_plot_all(df, keys_name=None):
    df = df.copy()
    df = df[df.HeaderHash == 'C8D67583']
    df = df.sort_values(by=['NumberOfKeys', 'BestCoverageType', 'HashFunction'])
    if keys_name:
        if isinstance(keys_name, list):
            s = ', '.join(f'"{k}"' for k in keys_name)
            df = df.query(f'KeysName in ({s})')
        else:
            df = df.query(f'KeysName == "{keys_name}"')
    timestamps = df.TableTimestamp.values
    results = []
    for t in timestamps:
        d = df.query(f'TableTimestamp == "{t}"')
        assert(len(d) == 1)
        #(figure, axes) = ridgeline_plot(d)
        #results.append((figure, axes))
        if d.NumberOfKeys.values[0] <= 5000:
            continue
        #print(d[['NumberOfKeys', 'BestCoverageType', 'HashFunction', 'TableTimestamp']].values[0])
        #ridgeline_plot(d)
        (figure, axes) = ridgeline_plot(d)
        #results.append((figure, axes))
    return results

def regplot_solutions_found(df):
    import matplotlib.pyplot as plt
    #plt.ioff()
    import seaborn as sns
    import numpy as np

    edges = [ 1 << i for i in range(12, 17) ]
    colors_array = plt.cm.tab20(np.arange(20).astype(int))
    np.random.shuffle(colors_array)
    color_map = { k: colors_array[i] for (i, k) in enumerate(edges) }

    fig = plt.figure(figsize=(10, 7))
    #edges = [ 1 << i for i in range(3, 17) ]
    keys = [
        'SolutionsFoundRatio',
        'KeysToEdgesRatio',
        'NumberOfEdges',
        'NumberOfKeys',
    ]
    for edge in edges:
        color = color_map[edge]
        #print((edge, color))
        data = df[df.NumberOfEdges == edge][keys]
        sns.regplot(
            x='KeysToEdgesRatio',
            y='SolutionsFoundRatio',
            #hue='NumberOfEdges',
            data=data,
            color=color,
            scatter_kws={
                's': 1,
                'alpha': 1.0,
            },
            line_kws={
                'linewidth': 0.1,
                'alpha': 1.0,
            },
        )

    plt.legend(labels=edges)
    plt.xlabel("Keys To Edges Ratio")
    plt.ylabel("Solutions Found Ratio")
    return (plt.gcf(), plt.gca())

def lmplot_solutions_found(df):
    import matplotlib.pyplot as plt
    #plt.ioff()
    import seaborn as sns
    import numpy as np

    edges = [ 1 << i for i in range(4, 17) ]
    colors_array = plt.cm.tab20(np.arange(20).astype(int))
    np.random.shuffle(colors_array)
    color_map = { k: colors_array[i] for (i, k) in enumerate(edges) }

    fig = plt.figure(figsize=(10, 7))
    edges = [ 1 << i for i in range(3, 17) ]
    keys = [
        'SolutionsFoundRatio',
        'KeysToEdgesRatio',
        'NumberOfEdges',
        'NumberOfKeys',
    ]
    sns.set(color_codes=True)
    #fig = plt.figure(figsize=(10, 7))
    keys = [
        'SolutionsFoundRatio',
        'KeysToEdgesRatio',
        'NumberOfEdges',
    ]
    sns.lmplot(
        x='KeysToEdgesRatio',
        y='SolutionsFoundRatio',
        hue='NumberOfEdges',
        col='NumberOfEdges',
        data=df[df.NumberOfEdges >= 16][keys],
        col_wrap=3,
        scatter_kws={'s': 1},
    )

    plt.legend(labels=edges)
    plt.xlabel("Keys To Edges Ratio")
    plt.ylabel("Solutions Found Ratio")
    return (plt.gcf(), plt.gca())

#===============================================================================
# Bokeh
#===============================================================================

def scatter6(df, min_num_edges=None, max_num_edges=None, p=None,
             show_plot=True, figure_kwds=None, circle_kwds=None):

    import numpy as np

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        TapTool,
        ColorBar,
        CustomJS,
        ColumnDataSource,
    )

    from bokeh.plotting import (
        figure,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    if figure_kwds is None:
        figure_kwds = {
            'tooltips': [
                ("Index", "@index"),
                ("Keys", "@KeysName"),
                ("Number of Keys", "@NumberOfKeys"),
                ("Number of Edges", "@NumberOfEdges"),
                ("Keys to Edges Ratio", "@KeysToEdgesRatio"),
                ("Solutions Found", "@SolutionsFoundRatio{(.000)}"),
            ],
        }

    if circle_kwds is None:
        circle_kwds = {}

    if not p:
        p = figure(
            plot_width=1000,
            plot_height=1000,
            tools='pan,wheel_zoom,box_select,lasso_select,reset,tap,hover',
            **figure_kwds,
        )


    if min_num_edges is None:
        min_num_edges = 256

    if max_num_edges is None:
        #max_num_edges = df.NumberOfEdges.max()
        max_num_edges = 65536

    df = df.query(f'NumberOfEdges >= {min_num_edges} and '
                  f'NumberOfEdges <= {max_num_edges}').copy()

    df['NumberOfEdgesStr'] = df.NumberOfEdges.values.astype(np.str)

    num_edges = [ 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 ]
    num_edges_str = [ str(e) for e in num_edges ]

    colors_map = { e: bp.Spectral11[i] for (i, e) in enumerate(num_edges) }

    color_mapper = bt.factor_cmap(
        field_name='NumberOfEdgesStr',
        palette=bp.Spectral11,
        factors=num_edges_str,
    )

    greys = [ bp.Greys256[i] for i in range(-50, 60, 10) ]
    greys_map = { e: greys[i] for (i, e) in enumerate(num_edges) }

    grey_mapper = bt.factor_cmap(
        field_name='NumberOfEdgesStr',
        palette=greys,
        factors=num_edges_str,
    )

    df['Color'] = [ colors_map[e] for e in df.NumberOfEdges.values ]

    source = ColumnDataSource(df)

    cr = p.circle(
        'KeysToEdgesRatio',
        'SolutionsFoundRatio',
        color='Color',
        line_color='Color',
        source=source,
        #fill_color='grey',
        #fill_alpha=0.05,
        #hover_fill_color=mapper,
        #hover_alpha=0.3,
        #line_color=None,
        #hover_line_color=mapper,
        #line_color=color_mapper,
        #color=color_mapper,
        legend_field='NumberOfEdgesStr',
        **circle_kwds,
    )

    #color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0,0))
    #p.add_layout(color_bar, 'right')

    #p.background_fill_color = "#eeeeee"
    #p.grid.grid_line_color = "white"

    #ht = HoverTool(tooltips=)
    #p.add_tools(HoverTool())

    code = """
        //console.log(cb_data);
        //console.log(cb_obj);
        //console.log(s);
        //console.log(greys_map);
        //console.log(colors_map);
        const d = s.data;
        const selected_index = s.selected.indices[0];
        const selected_num_edges = d['NumberOfEdges'][selected_index];
        const selected_color = colors_map[selected_num_edges];
        var num_edges;
        var color;

        console.log("Tap!");
        console.log(selected_index);
        //console.log(selected_num_edges);
        //console.log(selected_color);
        //console.log(s);
        //console.log(d.length);

        for (var i = 0; i < d['index'].length; i++) {

            num_edges = d['NumberOfEdges'][i];
            if (num_edges == selected_num_edges) {
                color = selected_color;
            } else {
                color = greys_map[selected_num_edges];
            }

            //console.log(i, color);
            d['Color'][i] = color;
        }

        s.change.emit();
    """

    args = {
        's': source,
        'greys_map': greys_map,
        'colors_map': colors_map,
    }

    callback = CustomJS(args=args, code=code)

    taptool = p.select(type=TapTool)
    taptool.callback = callback

    if show_plot:
        show(p)

    return p

def scatter7(df, min_num_edges=None, max_num_edges=None, p=None,
             show_plot=True, figure_kwds=None, circle_kwds=None):

    import numpy as np

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        TapTool,
        ColorBar,
        CustomJS,
        ColumnDataSource,
    )

    from bokeh.plotting import (
        figure,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    if figure_kwds is None:
        figure_kwds = {
            'tooltips': [
                ("Index", "@index"),
                ("Keys", "@KeysName"),
                ("Number of Keys", "@NumberOfKeys"),
                ("Number of Edges", "@NumberOfEdges"),
                ("Keys to Edges Ratio", "@KeysToEdgesRatio"),
                ("Solutions Found Ratio", "@SolutionsFoundRatio{(0.000)}"),
            ],
        }

    if circle_kwds is None:
        circle_kwds = {}

    if not p:
        p = figure(
            plot_width=1000,
            plot_height=1000,
            tools='pan,wheel_zoom,box_select,lasso_select,reset,tap,hover',
            **figure_kwds,
        )


    if min_num_edges is None:
        min_num_edges = 256

    if max_num_edges is None:
        #max_num_edges = df.NumberOfEdges.max()
        max_num_edges = 65536

    df = df.query(f'NumberOfEdges >= {min_num_edges} and '
                  f'NumberOfEdges <= {max_num_edges}').copy()

    df['NumberOfEdgesStr'] = df.NumberOfEdges.values.astype(np.str)

    num_edges = [ 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 ]
    num_edges_str = [ str(e) for e in num_edges ]

    colors_map = { e: bp.Spectral11[i] for (i, e) in enumerate(num_edges) }
    colors_map[8192] = "#ffff00"

    color_mapper = bt.factor_cmap(
        field_name='NumberOfEdgesStr',
        palette=bp.Spectral11,
        factors=num_edges_str,
    )

    greys = [ bp.Greys256[i] for i in range(-50, 60, 10) ]
    greys_map = { e: greys[i] for (i, e) in enumerate(num_edges) }

    grey_mapper = bt.factor_cmap(
        field_name='NumberOfEdgesStr',
        palette=greys,
        factors=num_edges_str,
    )

    df['Color'] = [ colors_map[e] for e in df.NumberOfEdges.values ]

    source = ColumnDataSource(df)

    cr = p.circle(
        'KeysToEdgesRatio',
        'SolutionsFoundRatio',
        color='Color',
        line_color='Color',
        source=source,
        #fill_color='grey',
        #fill_alpha=0.05,
        #hover_fill_color=mapper,
        #hover_alpha=0.3,
        #line_color=None,
        #hover_line_color=mapper,
        #line_color=color_mapper,
        #color=color_mapper,
        legend_field='NumberOfEdgesStr',
        **circle_kwds,
    )

    #color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0,0))
    #p.add_layout(color_bar, 'right')

    p.background_fill_color = "#eeeeee"
    p.grid.grid_line_color = "white"

    #ht = HoverTool(tooltips=)
    #p.add_tools(HoverTool())

    code = """
        //console.log(cb_data);
        //console.log(cb_obj);
        //console.log(s);
        //console.log(greys_map);
        //console.log(colors_map);
        const d = s.data;
        const selected_index = s.selected.indices[0];
        const selected_num_edges = d['NumberOfEdges'][selected_index];
        const selected_color = colors_map[selected_num_edges];
        var num_edges;
        var color;
        var new_selection = [];

        //console.log("Tap!");
        //console.log(selected_index);
        //console.log(selected_num_edges);
        //console.log(selected_color);
        //console.log(s);
        //console.log(d.length);

        for (var i = 0; i < d['index'].length; i++) {

            num_edges = d['NumberOfEdges'][i];
            if (num_edges == selected_num_edges) {
                new_selection.push(i);
            }

            //console.log(i, color);
            //d['Color'][i] = color;
        }

        //console.log(new_selection);
        s.selected.indices = new_selection;

        s.change.emit();
    """

    args = {
        's': source,
        'greys_map': greys_map,
        'colors_map': colors_map,
    }

    callback = CustomJS(args=args, code=code)

    taptool = p.select(type=TapTool)
    taptool.callback = callback

    if show_plot:
        show(p)

    return p

def panel1(df, min_num_edges=None, max_num_edges=None,
           show_plot=True, figure_kwds=None, circle_kwds=None):

    import numpy as np
    import pandas as pd

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        Tabs,
        Panel,
        Select,
        TapTool,
        Range1d,
        ColorBar,
        CustomJS,
        RangeSlider,
        MultiSelect,
        ColumnDataSource,
        RadioButtonGroup,
    )

    from bokeh.plotting import (
        figure,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    tooltips = [
        ("Index", "@index"),
        ("Keys", "@KeysName"),
        ("Number of Keys", "@NumberOfKeys"),
        ("Number of Edges", "@NumberOfEdges"),
        ("Keys to Edges Ratio", "@KeysToEdgesRatio"),
        ("Solutions Found Ratio", "@SolutionsFoundRatio{(0.000)}"),
    ]

    if figure_kwds is None:
        figure_kwds = {}

    if 'tooltips' not in figure_kwds:
        figure_kwds['tooltips'] = tooltips

    if circle_kwds is None:
        circle_kwds = {}

    if min_num_edges is None:
        min_num_edges = 256

    if max_num_edges is None:
        #max_num_edges = df.NumberOfEdges.max()
        max_num_edges = 65536

    # Prep colors.
    num_edges = [ 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 ]
    num_edges_str = [ str(e) for e in num_edges ]

    colors_map = { e: bp.Spectral11[i] for (i, e) in enumerate(num_edges) }
    # Change the light yellow color to a darker color.
    colors_map[8192] = "#f4d03f"

    line_colors_map = { e: bp.Spectral11[i] for (i, e) in enumerate(num_edges) }
    line_colors_map[8192] = "#f1c40f"

    color_mapper = bt.factor_cmap(
        field_name='NumberOfEdgesStr',
        palette=bp.Spectral11,
        factors=num_edges_str,
    )

    greys = [ bp.Greys256[i] for i in range(-50, 60, 10) ]
    greys_map = { e: greys[i] for (i, e) in enumerate(num_edges) }

    grey_mapper = bt.factor_cmap(
        field_name='NumberOfEdgesStr',
        palette=greys,
        factors=num_edges_str,
    )

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    panels = []

    source_df = df

    y_range = Range1d(0, 1.0)

    for hash_func in hash_funcs:

        df = source_df.query(f'NumberOfEdges >= {min_num_edges} and '
                             f'NumberOfEdges <= {max_num_edges} and '
                             f'HashFunction == "{hash_func}"').copy()

        df['NumberOfEdgesStr'] = df.NumberOfEdges.values.astype(np.str)

        df['Color'] = [ colors_map[e] for e in df.NumberOfEdges.values ]
        df['LineColor'] = [
            line_colors_map[e] for e in df.NumberOfEdges.values
        ]

        df['KeysToEdgesRatio'] = np.around(df.KeysToEdgesRatio.values, 3)
        df['SolutionsFoundRatio'] = np.around(df.SolutionsFoundRatio.values, 3)

        sdf = (
            df.groupby(['KeysToEdgesRatio', 'SolutionsFoundRatio'])
              .size()
              .reset_index()
              .rename(columns={0: 'Size'})
        )

        sdf['LogSize'] = np.log(sdf['Size'].values * 2)

        size_map = {}

        for (i, row) in sdf.iterrows():
            k = (row.KeysToEdgesRatio, row.SolutionsFoundRatio)
            v = (row.Size, row.LogSize)

            size_map[k] = v

        df['Size'] = np.float(0)
        df['LogSize'] = np.float(0)

        for (i, row) in df.iterrows():
            k = (row.KeysToEdgesRatio, row.SolutionsFoundRatio)
            (size, log_size) = size_map[k]
            df.at[i, 'Size'] = size * 3
            df.at[i, 'LogSize'] = log_size

        source = ColumnDataSource(df)

        p = figure(
            plot_width=1000,
            plot_height=1000,
            tools='pan,wheel_zoom,box_select,lasso_select,reset,tap,hover',
            **figure_kwds,
        )

        p.y_range = y_range

        cr = p.circle(
            'KeysToEdgesRatio',
            'SolutionsFoundRatio',
            color='Color',
            size='Size',
            fill_alpha=0.5,
            line_alpha=1.0,
            line_color='LineColor',
            source=source,
            legend_field='NumberOfEdgesStr',
            **circle_kwds,
        )

        p.background_fill_color = "#eeeeee"
        p.grid.grid_line_color = "white"

        code = """
            const d = s.data;
            const selected_index = s.selected.indices[0];
            const selected_num_edges = d['NumberOfEdges'][selected_index];
            const selected_color = colors_map[selected_num_edges];
            var num_edges;
            var color;
            var new_selection = [];

            for (var i = 0; i < d['index'].length; i++) {
                num_edges = d['NumberOfEdges'][i];
                if (num_edges == selected_num_edges) {
                    new_selection.push(i);
                }
            }

            s.selected.indices = new_selection;

            s.change.emit();
        """

        args = {
            's': source,
            'greys_map': greys_map,
            'colors_map': colors_map,
        }

        callback = CustomJS(args=args, code=code)

        taptool = p.select(type=TapTool)
        taptool.callback = callback

        panel = Panel(child=p, title=hash_func)
        panels.append(panel)

    tabs = Tabs(tabs=panels)

    if show_plot:
        show(tabs)

    return p

def grid1(df, min_num_edges=None, max_num_edges=None,
           show_plot=True, figure_kwds=None, circle_kwds=None):

    import numpy as np
    import pandas as pd

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        Tabs,
        Panel,
        Select,
        TapTool,
        Range1d,
        ColorBar,
        CustomJS,
        RangeSlider,
        MultiSelect,
        ColumnDataSource,
        RadioButtonGroup,
    )

    from bokeh.plotting import (
        figure,
    )

    from bokeh.layouts import (
        gridplot,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    tooltips = [
        ("Index", "@index"),
        ("Keys", "@KeysName"),
        ("Number of Keys", "@NumberOfKeys"),
        ("Number of Edges", "@NumberOfEdges"),
        ("Keys to Edges Ratio", "@KeysToEdgesRatio"),
        ("Solutions Found Ratio", "@SolutionsFoundRatio{(0.000)}"),
    ]

    if figure_kwds is None:
        figure_kwds = {}

    if 'tooltips' not in figure_kwds:
        figure_kwds['tooltips'] = tooltips

    if circle_kwds is None:
        circle_kwds = {}

    if min_num_edges is None:
        min_num_edges = 256

    if max_num_edges is None:
        #max_num_edges = df.NumberOfEdges.max()
        max_num_edges = 65536

    # Prep colors.
    num_edges = [ 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 ]
    num_edges_str = [ str(e) for e in num_edges ]

    colors_map = { e: bp.Spectral11[i] for (i, e) in enumerate(num_edges) }
    # Change the light yellow color to a darker color.
    colors_map[8192] = "#f4d03f"

    line_colors_map = { e: bp.Spectral11[i] for (i, e) in enumerate(num_edges) }
    line_colors_map[8192] = "#f1c40f"

    color_mapper = bt.factor_cmap(
        field_name='NumberOfEdgesStr',
        palette=bp.Spectral11,
        factors=num_edges_str,
    )

    greys = [ bp.Greys256[i] for i in range(-50, 60, 10) ]
    greys_map = { e: greys[i] for (i, e) in enumerate(num_edges) }

    grey_mapper = bt.factor_cmap(
        field_name='NumberOfEdgesStr',
        palette=greys,
        factors=num_edges_str,
    )

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    figures = []

    source_df = df

    y_range = Range1d(0, 1.0)

    for hash_func in hash_funcs:

        df = source_df.query(f'NumberOfEdges >= {min_num_edges} and '
                             f'NumberOfEdges <= {max_num_edges} and '
                             f'HashFunction == "{hash_func}"').copy()

        df['NumberOfEdgesStr'] = df.NumberOfEdges.values.astype(np.str)

        df['Color'] = [ colors_map[e] for e in df.NumberOfEdges.values ]
        df['LineColor'] = [
            line_colors_map[e] for e in df.NumberOfEdges.values
        ]

        df['KeysToEdgesRatio'] = np.around(df.KeysToEdgesRatio.values, 3)
        df['SolutionsFoundRatio'] = np.around(df.SolutionsFoundRatio.values, 3)

        sdf = (
            df.groupby(['KeysToEdgesRatio', 'SolutionsFoundRatio'])
              .size()
              .reset_index()
              .rename(columns={0: 'Size'})
        )

        sdf['LogSize'] = np.log(sdf['Size'].values * 2)

        size_map = {}

        for (i, row) in sdf.iterrows():
            k = (row.KeysToEdgesRatio, row.SolutionsFoundRatio)
            v = (row.Size, row.LogSize)

            size_map[k] = v

        df['Size'] = np.float(0)
        df['LogSize'] = np.float(0)

        for (i, row) in df.iterrows():
            k = (row.KeysToEdgesRatio, row.SolutionsFoundRatio)
            (size, log_size) = size_map[k]
            df.at[i, 'Size'] = size * 3
            df.at[i, 'LogSize'] = log_size

        source = ColumnDataSource(df)

        p = figure(
            plot_width=500,
            plot_height=500,
            tools='pan,wheel_zoom,box_select,lasso_select,reset,tap,hover',
            **figure_kwds,
        )

        p.y_range = y_range

        cr = p.circle(
            'KeysToEdgesRatio',
            'SolutionsFoundRatio',
            color='Color',
            size='Size',
            fill_alpha=0.5,
            line_alpha=1.0,
            line_color='LineColor',
            source=source,
            legend_field='NumberOfEdgesStr',
            **circle_kwds,
        )

        p.background_fill_color = "#eeeeee"
        p.grid.grid_line_color = "white"

        code = """
            const d = s.data;
            const selected_index = s.selected.indices[0];
            const selected_num_edges = d['NumberOfEdges'][selected_index];
            const selected_color = colors_map[selected_num_edges];
            var num_edges;
            var color;
            var new_selection = [];

            for (var i = 0; i < d['index'].length; i++) {
                num_edges = d['NumberOfEdges'][i];
                if (num_edges == selected_num_edges) {
                    new_selection.push(i);
                }
            }

            s.selected.indices = new_selection;

            s.change.emit();
        """

        args = {
            's': source,
            'greys_map': greys_map,
            'colors_map': colors_map,
        }

        callback = CustomJS(args=args, code=code)

        taptool = p.select(type=TapTool)
        taptool.callback = callback

        figures.append(p)

    figures.insert(1, None)
    grid = gridplot(figures, ncols=4)

    if show_plot:
        show(grid)

    return p

def grid2(df, min_num_edges=None, max_num_edges=None,
           show_plot=True, figure_kwds=None, circle_kwds=None,
           color_category=None):

    import numpy as np
    import pandas as pd

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        Tabs,
        Panel,
        Select,
        TapTool,
        Range1d,
        ColorBar,
        CustomJS,
        RangeSlider,
        MultiSelect,
        ColumnDataSource,
        RadioButtonGroup,
    )

    from bokeh.plotting import (
        figure,
    )

    from bokeh.layouts import (
        gridplot,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    tooltips = [
        ("Index", "@index"),
        ("Keys", "@KeysName"),
        ("Number of Keys", "@NumberOfKeys"),
        ("Number of Edges", "@NumberOfEdges"),
        ("Keys to Edges Ratio", "@KeysToEdgesRatio"),
        ("Solutions Found Ratio", "@SolutionsFoundRatio{(0.000)}"),
    ]

    if figure_kwds is None:
        figure_kwds = {}

    if 'tooltips' not in figure_kwds:
        figure_kwds['tooltips'] = tooltips

    if circle_kwds is None:
        circle_kwds = {}

    if min_num_edges is None:
        min_num_edges = 256

    if max_num_edges is None:
        max_num_edges = df.NumberOfEdges.max()

    if color_category is None:
        color_category = bp.Category20

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    figures = []

    source_df = df

    y_range = Range1d(0, 1.0)

    for hash_func in hash_funcs:

        df = source_df.query(f'NumberOfEdges >= {min_num_edges} and '
                             f'NumberOfEdges <= {max_num_edges} and '
                             f'HashFunction == "{hash_func}"').copy()

        df['NumberOfEdgesStr'] = df.NumberOfEdges.values.astype(np.str)

        num_edges = (
            df.NumberOfEdges
                .value_counts()
                .sort_index()
                .index
                .values
        )
        num_edges_str = [ str(e) for e in num_edges ]

        if isinstance(color_category, dict):
            cat = color_category[len(num_edges)]
        else:
            cat = color_category

        colors_map = { e: cat[i] for (i, e) in enumerate(num_edges_str) }

        df['Color'] = [ colors_map[e] for e in df.NumberOfEdgesStr.values ]

        df['KeysToEdgesRatio'] = np.around(df.KeysToEdgesRatio.values, 3)
        df['SolutionsFoundRatio'] = np.around(df.SolutionsFoundRatio.values, 3)

        sdf = (
            df.groupby(['KeysToEdgesRatio', 'SolutionsFoundRatio'])
              .size()
              .reset_index()
              .rename(columns={0: 'Size'})
        )

        sdf['LogSize'] = np.log(sdf['Size'].values * 2)

        size_map = {}

        for (i, row) in sdf.iterrows():
            k = (row.KeysToEdgesRatio, row.SolutionsFoundRatio)
            v = (row.Size, row.LogSize)

            size_map[k] = v

        df['Size'] = np.float(0)
        df['LogSize'] = np.float(0)

        for (i, row) in df.iterrows():
            k = (row.KeysToEdgesRatio, row.SolutionsFoundRatio)
            (size, log_size) = size_map[k]
            df.at[i, 'Size'] = size * 3
            df.at[i, 'LogSize'] = log_size

        df.sort_values(by=['NumberOfEdges'])

        source = ColumnDataSource(df)

        p = figure(
            plot_width=500,
            plot_height=500,
            tools='pan,wheel_zoom,box_select,lasso_select,reset,tap,hover',
            **figure_kwds,
        )

        p.y_range = y_range

        cr = p.circle(
            'KeysToEdgesRatio',
            'SolutionsFoundRatio',
            color='Color',
            size='Size',
            fill_alpha=0.5,
            line_alpha=1.0,
            source=source,
            legend_field='NumberOfEdgesStr',
            **circle_kwds,
        )

        p.background_fill_color = "#eeeeee"
        p.grid.grid_line_color = "white"

        code = """
            const d = s.data;
            const selected_index = s.selected.indices[0];
            const selected_num_edges = d['NumberOfEdges'][selected_index];
            var num_edges;
            var new_selection = [];

            for (var i = 0; i < d['index'].length; i++) {
                num_edges = d['NumberOfEdges'][i];
                if (num_edges == selected_num_edges) {
                    new_selection.push(i);
                }
            }

            s.selected.indices = new_selection;

            s.change.emit();
        """

        args = {
            's': source,
            'colors_map': colors_map,
        }

        callback = CustomJS(args=args, code=code)

        taptool = p.select(type=TapTool)
        taptool.callback = callback

        figures.append(p)

    figures.insert(1, None)
    grid = gridplot(figures, ncols=4)

    if show_plot:
        show(grid)

    return p

def grid3(df, lrdf=None, min_num_vertices=None, max_num_vertices=None,
          show_plot=True, figure_kwds=None, circle_kwds=None,
          color_category=None, ncols=None):

    import numpy as np
    import pandas as pd

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        Tabs,
        Panel,
        Select,
        TapTool,
        Range1d,
        ColorBar,
        CustomJS,
        RangeSlider,
        MultiSelect,
        ColumnDataSource,
        RadioButtonGroup,
    )

    from bokeh.plotting import (
        figure,
    )

    from bokeh.layouts import (
        gridplot,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    tooltips = [
        ("Index", "@index"),
        ("Keys", "@KeysName"),
        ("Number of Keys", "@NumberOfKeys"),
        ("Number of Vertices", "@NumberOfVertices"),
        ("Keys to Edges Ratio", "@KeysToEdgesRatio"),
        ("Keys to Vertices Ratio", "@KeysToVerticesRatio"),
        ("Solutions Found Ratio", "@SolutionsFoundRatio{(0.000)}"),
    ]

    if figure_kwds is None:
        figure_kwds = {}

    if 'tooltips' not in figure_kwds:
        figure_kwds['tooltips'] = tooltips

    if circle_kwds is None:
        circle_kwds = {}

    if min_num_vertices is None:
        min_num_vertices = 256

    if max_num_vertices is None:
        max_num_vertices = df.NumberOfVertices.max()

    if color_category is None:
        color_category = bp.Category20

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    figures = []

    source_df = df

    y_range = Range1d(0, 1.0)

    for hash_func in hash_funcs:

        df = source_df.query(f'NumberOfVertices >= {min_num_vertices} and '
                             f'NumberOfVertices <= {max_num_vertices} and '
                             f'HashFunction == "{hash_func}"').copy()

        df['NumberOfVerticesStr'] = df.NumberOfVertices.values.astype(np.str)

        num_vertices = (
            df.NumberOfVertices
                .value_counts()
                .sort_index()
                .index
                .values
        )
        num_vertices_str = [ str(e) for e in num_vertices ]

        if isinstance(color_category, dict):
            cat = color_category[len(num_vertices)]
        else:
            cat = color_category

        colors_map = { e: cat[i] for (i, e) in enumerate(num_vertices_str) }

        df['Color'] = [ colors_map[e] for e in df.NumberOfVerticesStr.values ]

        df['KeysToVerticesRatio'] = np.around(df.KeysToVerticesRatio.values, 3)
        df['SolutionsFoundRatio'] = np.around(df.SolutionsFoundRatio.values, 3)

        sdf = (
            df.groupby(['KeysToVerticesRatio', 'SolutionsFoundRatio'])
              .size()
              .reset_index()
              .rename(columns={0: 'Size'})
        )

        sdf['LogSize'] = np.log(sdf['Size'].values * 2)

        size_map = {}

        for (i, row) in sdf.iterrows():
            k = (row.KeysToVerticesRatio, row.SolutionsFoundRatio)
            v = (row.Size, row.LogSize)

            size_map[k] = v

        df['Size'] = np.float(0)
        df['LogSize'] = np.float(0)

        for (i, row) in df.iterrows():
            k = (row.KeysToVerticesRatio, row.SolutionsFoundRatio)
            (size, log_size) = size_map[k]
            df.at[i, 'Size'] = size * 3
            df.at[i, 'LogSize'] = log_size

        df.sort_values(by=['NumberOfVertices'])

        source = ColumnDataSource(df)

        p = figure(
            plot_width=500,
            plot_height=500,
            tools='pan,wheel_zoom,box_select,lasso_select,reset,tap,hover',
            **figure_kwds,
        )

        p.title.text = hash_func
        p.y_range = y_range
        p.xaxis.axis_label = 'Keys to Vertices Ratio'
        p.yaxis.axis_label = 'Probability of Finding Solution'

        cr = p.circle(
            'KeysToVerticesRatio',
            'SolutionsFoundRatio',
            color='Color',
            size='Size',
            fill_alpha=0.5,
            line_alpha=1.0,
            source=source,
            legend_group='NumberOfVertices',
            **circle_kwds,
        )

        p.background_fill_color = "#eeeeee"
        p.grid.grid_line_color = "white"

        code = """
            const d = s.data;
            const selected_index = s.selected.indices[0];
            const selected_num_vertices = d['NumberOfVertices'][selected_index];
            var num_vertices;
            var new_selection = [];

            for (var i = 0; i < d['index'].length; i++) {
                num_vertices = d['NumberOfVertices'][i];
                if (num_vertices == selected_num_vertices) {
                    new_selection.push(i);
                }
            }

            s.selected.indices = new_selection;

            s.change.emit();
        """

        args = {
            's': source,
            'colors_map': colors_map,
        }

        callback = CustomJS(args=args, code=code)

        taptool = p.select(type=TapTool)
        taptool.callback = callback

        figures.append(p)

    if not ncols:
        ncols = 2

    #figures.insert(1, None)
    grid = gridplot(figures, ncols=ncols)

    if show_plot:
        show(grid)

    return p

def grid4(df, lrdf=None, min_num_edges=None, max_num_edges=None,
          show_plot=True, figure_kwds=None, circle_kwds=None,
          color_category=None, ncols=None):

    import numpy as np
    import pandas as pd

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        Tabs,
        Panel,
        Select,
        TapTool,
        Range1d,
        ColorBar,
        CustomJS,
        RangeSlider,
        MultiSelect,
        ColumnDataSource,
        RadioButtonGroup,
    )

    from bokeh.plotting import (
        figure,
    )

    from bokeh.layouts import (
        gridplot,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    tooltips = [
        ("Index", "@index"),
        ("Keys", "@KeysName"),
        ("Number of Keys", "@NumberOfKeys"),
        ("Number of Edges", "@NumberOfEdges"),
        ("Number of Vertices", "@NumberOfVertices"),
        ("Number Of Resizes", "@NumberOfTableResizeEvents"),
        ("Keys to Edges Ratio", "@KeysToEdgesRatio"),
        ("Keys to Vertices Ratio", "@KeysToVerticesRatio"),
        ("Solutions Found Ratio", "@SolutionsFoundRatio{(0.000)}"),
    ]

    if figure_kwds is None:
        figure_kwds = {}

    if 'tooltips' not in figure_kwds:
        figure_kwds['tooltips'] = tooltips

    if circle_kwds is None:
        circle_kwds = {}

    if min_num_edges is None:
        min_num_edges = 256

    if max_num_edges is None:
        max_num_edges = df.NumberOfEdges.max()

    if color_category is None:
        color_category = bp.Spectral11

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    figures = []

    source_df = df

    y_range = Range1d(0, 1.0)

    for hash_func in hash_funcs:

        df = source_df.query(f'NumberOfEdges >= {min_num_edges} and '
                             f'NumberOfEdges <= {max_num_edges} and '
                             f'HashFunction == "{hash_func}"').copy()

        df['NumberOfEdgesStr'] = df.NumberOfEdges.values.astype(np.str)

        num_edges = (
            df.NumberOfEdges
                .value_counts()
                .sort_index()
                .index
                .values
        )
        num_edges_str = [ str(e) for e in num_edges ]

        if isinstance(color_category, dict):
            cat = color_category[len(num_edges)]
        else:
            cat = color_category

        colors_map = { e: cat[i] for (i, e) in enumerate(num_edges_str) }

        df['Color'] = [ colors_map[e] for e in df.NumberOfEdgesStr.values ]

        df['KeysToEdgesRatio'] = np.around(df.KeysToEdgesRatio.values, 3)
        df['SolutionsFoundRatio'] = np.around(df.SolutionsFoundRatio.values, 3)

        sdf = (
            df.groupby(['KeysToEdgesRatio', 'SolutionsFoundRatio'])
              .size()
              .reset_index()
              .rename(columns={0: 'Size'})
        )

        sdf['LogSize'] = np.log(sdf['Size'].values * 2)

        size_map = {}

        for (i, row) in sdf.iterrows():
            k = (row.KeysToEdgesRatio, row.SolutionsFoundRatio)
            v = (row.Size, row.LogSize)

            size_map[k] = v

        df['Size'] = np.float(0)
        df['LogSize'] = np.float(0)

        for (i, row) in df.iterrows():
            k = (row.KeysToEdgesRatio, row.SolutionsFoundRatio)
            (size, log_size) = size_map[k]
            df.at[i, 'Size'] = size * 3
            df.at[i, 'LogSize'] = log_size

        df.sort_values(by=['NumberOfEdges'])

        p = figure(
            plot_width=500,
            plot_height=500,
            tools='pan,wheel_zoom,box_select,lasso_select,reset,tap,hover',
            **figure_kwds,
        )

        p.title.text = hash_func
        p.y_range = y_range
        p.xaxis.axis_label = 'Keys to Edges Ratio'
        p.yaxis.axis_label = 'Probability of Finding Solution'

        source = ColumnDataSource(df)

        cr = p.circle(
            'KeysToEdgesRatio',
            'SolutionsFoundRatio',
            color='Color',
            size='Size',
            fill_alpha=0.5,
            line_alpha=1.0,
            source=source,
            legend_group='NumberOfEdges',
            **circle_kwds,
        )

        p.background_fill_color = "#eeeeee"
        p.grid.grid_line_color = "white"

        code = """
            const d = s.data;
            const selected_index = s.selected.indices[0];
            const selected_num_edges = d['NumberOfEdges'][selected_index];
            var num_edges;
            var new_selection = [];

            for (var i = 0; i < d['index'].length; i++) {
                num_edges = d['NumberOfEdges'][i];
                if (num_edges == selected_num_edges) {
                    new_selection.push(i);
                }
            }

            s.selected.indices = new_selection;

            s.change.emit();
        """

        args = {
            's': source,
            'colors_map': colors_map,
        }

        callback = CustomJS(args=args, code=code)

        taptool = p.select(type=TapTool)
        taptool.callback = callback

        figures.append(p)

    if not ncols:
        ncols = 2

    #figures.insert(1, None)
    grid = gridplot(figures, ncols=ncols)

    if show_plot:
        show(grid)

    return p


# vim:set ts=8 sw=4 sts=4 tw=80 et                                             :
