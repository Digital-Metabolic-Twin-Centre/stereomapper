""" Script to query the database and map the results to the original data. This will map the score results to the original data, which can then be used for ROC analysis. """

import pandas as pd
from pathlib import Path
import sqlite3

def make_undirected_pairkey(row: pd.Series) -> str:
    """
    Create an undirected pairkey.

    This treats A_B and B_A as the same molecular pair.
    """
    id1 = str(row["id1"])
    id2 = str(row["id2"])

    return "_".join(sorted([id1, id2]))

def query_for_relationships(input_db_path, output_csv_path):
    """
    Query the database for the relationships between the pairs in the benchmarking datasets, and save the results to a CSV file.
    """

    conn = sqlite3.connect(input_db_path)
    query = """
    SELECT 
    cluster_a_members,
    cluster_b_members,
    classification,
    score
    from relationships 
    where classification <> 'Unclassified'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df 

def extract_identifiers(df):

    """
    Extracts the identifiers from JSON strings in the cluster_a_members and cluster_b_members columns, and adds them as new columns id1 and id2.
    """
    df['id1'] = df['cluster_a_members'].apply(lambda x: eval(x)[0] if isinstance(x, str) else None)
    df['id2'] = df['cluster_b_members'].apply(lambda x: eval(x)[0] if isinstance(x, str) else None)

    # strip local: prefix if it exists
    df['id1'] = df['id1'].apply(lambda x: x.replace("local:", "") if isinstance(x, str) else x)
    df['id2'] = df['id2'].apply(lambda x: x.replace("local:", "") if isinstance(x, str) else x)

    # normalise chebi ids
    df['id1'] = df['id1'].apply(lambda x: x.replace("chebi:", "CHEBI_") if isinstance(x, str) else x)
    df['id2'] = df['id2'].apply(lambda x: x.replace("chebi:", "CHEBI_") if isinstance(x, str) else x)

    df['predicted_class'] = df['classification']
    df['score_S'] = df['score']

    # drop cluster_a_members and cluster_b_members columns
    df = df.drop(columns=['cluster_a_members', 'cluster_b_members', 'classification', 'score'])

    return df

def map_scores_to_original_data(df, original_data_path):
    """
    Map the scores to the original data, using the pairkey to merge the dataframes. 
    
    This will add the predicted_class and score_S columns to the original dataframe, which can then be used for confidence score ROC evaluation.
    """
    original_df = pd.read_csv(original_data_path)

    # merge on pairkey
    merged_df = pd.merge(original_df, df[['pairkey', 'predicted_class', 'score_S']], on='pairkey', how='left')

    # # rename pairkey to pair_id 
    merged_df = merged_df.rename(columns={'pairkey': 'pair_id'})
    # merged_df = merged_df.drop(columns=['pair_id'])

    merged_df = merged_df.rename(columns={'true_label': 'true_class'})
    merged_df = merged_df.dropna(subset=['predicted_class', 'score_S'])

    return merged_df

def main():
    input_db_path = "/home/jackmcgoldrick/repos/stereomapper_original_release/experiments/manuscript_revision/roc_analysis/results/full_benchmark_for_conf_Score_thresholds.sqlite"
    output_csv_path = "/home/jackmcgoldrick/repos/stereomapper_original_release/experiments/manuscript_revision/roc_analysis/results/all_benchmarking_data_for_threshold_analysis.csv"
    original_csv_path = "/home/jackmcgoldrick/repos/stereomapper_original_release/experiments/manuscript_revision/roc_analysis/data/combined_relationship_benchmarking_dataset_filtered.csv"


    df = query_for_relationships(input_db_path, output_csv_path)
    #print(df.head(20))
    df = extract_identifiers(df)
    #print(df.head(20))
    df['pairkey'] = df.apply(make_undirected_pairkey, axis=1)

    #print(df.head(20))

    # map scores to original data
    mapped_df = map_scores_to_original_data(df, original_csv_path)
    print(mapped_df.head(20))

    # save to csv
    mapped_df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":    main()