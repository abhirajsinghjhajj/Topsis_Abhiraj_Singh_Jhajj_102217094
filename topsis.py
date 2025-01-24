import pandas as pd
import numpy as np
import sys

def check_inputs(input_file, weights, impacts, result_file):
    if not all([input_file, weights, impacts, result_file]):
        raise ValueError("Insufficient number of parameters. Required: input_file, weights, impacts, result_file")
    
    try:
        df = pd.read_csv(input_file)
        
        if len(df.columns) < 3:
            raise ValueError("Input file must contain three or more columns")
        
        num_criteria = len(df.columns) - 1
        
        if not df.iloc[:, 1:].applymap(np.isreal).all().all():
            raise ValueError("Columns from 2nd onwards must contain numeric values only")
        
        try:
            weights = [float(w.strip()) for w in weights.split(',')]
        except ValueError:
            raise ValueError(f"Invalid weights format. Expected {num_criteria} comma-separated numbers")
            
        try:
            impacts = [imp.strip() for imp in impacts.split(',')]
        except ValueError:
            raise ValueError(f"Invalid impacts format. Expected {num_criteria} comma-separated '+' or '-' symbols")
        
        if len(weights) != num_criteria:
            raise ValueError(f"Number of weights ({len(weights)}) should match number of criteria ({num_criteria})")
        
        if len(impacts) != num_criteria:
            raise ValueError(f"Number of impacts ({len(impacts)}) should match number of criteria ({num_criteria})")
        
        if not all(imp in ['+', '-'] for imp in impacts):
            raise ValueError("Impacts must be either +ve or -ve")
            
        return df, weights, impacts
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file '{input_file}' not found")

def topsis(df, weights, impacts):
    decision_matrix = df.iloc[:, 1:].values.astype(float)
    
    norm = np.sqrt(np.sum(decision_matrix**2, axis=0))
    normalized_matrix = decision_matrix / norm
    
    weighted_normalized = normalized_matrix * weights
    
    ideal_best = np.zeros(len(weights))
    ideal_worst = np.zeros(len(weights))
    
    for i in range(len(weights)):
        if impacts[i] == '+':
            ideal_best[i] = np.max(weighted_normalized[:, i])
            ideal_worst[i] = np.min(weighted_normalized[:, i])
        else:
            ideal_best[i] = np.min(weighted_normalized[:, i])
            ideal_worst[i] = np.max(weighted_normalized[:, i])
    
    s_best = np.sqrt(np.sum((weighted_normalized - ideal_best)**2, axis=1))
    s_worst = np.sqrt(np.sum((weighted_normalized - ideal_worst)**2, axis=1))
    
    topsis_score = s_worst / (s_best + s_worst)
    
    ranks = len(topsis_score) - np.argsort(topsis_score).argsort()
    
    return topsis_score, ranks

def main():
    if len(sys.argv) != 5:
        print("Incorrect Usage! Please use:")
        print("python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFileName>")
        print("Example: python 102217094.py 102217094-data.csv \"1,1,1,1,1\" \"+,+,-,+,+\" 102217094-result.csv")
        sys.exit(1)
        
    try:
        input_file = sys.argv[1]
        weights = sys.argv[2]
        impacts = sys.argv[3]
        result_file = sys.argv[4]
        
        df, processed_weights, processed_impacts = check_inputs(input_file, weights, impacts, result_file)
        
        scores, ranks = topsis(df, processed_weights, processed_impacts)
        
        df['Topsis Score'] = scores
        df['Rank'] = ranks
        
        df.to_csv(result_file, index=False)
        print(f"Results have been saved to {result_file}")
        
        num_criteria = len(df.columns) - 1
        print(f"\nNote: Your input file has {num_criteria} criteria (columns excluding the first).")
        print(f"Make sure to provide {num_criteria} weights and {num_criteria} impacts.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
