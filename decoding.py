import os
import h5py
import numpy as np
import pandas as pd
import re
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from collections import Counter
import pickle
import logging

def extract_odor_events(event_array, window='default'):
    """
    Extract odor events from event array with pattern identification.
    
    Args:
        event_array: List of event strings
        window: Specifies the time window to extract:
               'default': start_idx+1 to end_idx
               'pre_odor': start_idx+1 to end_idx-2
               'post_odor': start_idx+6 to end_idx
        
    Returns:
        Dictionary of events and tone patterns (ascending=0, descending=1)
    """
    result = {}
    tone_patterns = []
    pattern_ids = {
        # Ascending pattern gets ID 0
        'ascending': 0,
        # Descending pattern gets ID 1
        'descending': 1
    }
    i = 0
    event_id = 0

    # Regex patterns for event identification
    odor_pattern = re.compile(r'^Odor_(\d+)$')
    tone_on_pattern = re.compile(r'^TONE_ON_(\d+)Hz$')
    tone_off_pattern = re.compile(r'^TONE_OFF_(\d+)Hz$')

    while i < len(event_array):
        if event_array[i].startswith('ODOR_POKE'):
            start_idx = i
            i += 1
            early_unpoke = False

            # Search for the next valid ODOR_UNPOKE
            while i < len(event_array):
                if event_array[i].startswith('ODOR_UNPOKE_EARLY'):
                    early_unpoke = True
                    break
                elif event_array[i].startswith('ODOR_UNPOKE'):
                    end_idx = i
                    break
                i += 1
            else:
                break  # No ODOR_UNPOKE found

            if early_unpoke:
                i = start_idx + 1
                continue

            direction_idx = end_idx + 1 if end_idx + 1 < len(event_array) else None
            direction = event_array[direction_idx] if direction_idx is not None else None

            # Extract odor
            odor = None
            for j in range(start_idx, end_idx + 1):
                match = odor_pattern.match(event_array[j])
                if match:
                    odor = event_array[j]
                    break

            # Extract tone frequencies and determine if pattern is ascending or descending
            tone_frequencies = []
            for j in range(start_idx, end_idx + 1):
                match = tone_on_pattern.match(event_array[j])
                if match:
                    freq = int(match.group(1))
                    tone_frequencies.append(freq)
            
            # Determine pattern type (ascending or descending)
            tone_pattern_id = None
            if tone_frequencies:
                is_ascending = all(tone_frequencies[i] <= tone_frequencies[i+1] for i in range(len(tone_frequencies)-1))
                is_descending = all(tone_frequencies[i] >= tone_frequencies[i+1] for i in range(len(tone_frequencies)-1))
                
                if is_ascending:
                    tone_pattern_id = pattern_ids['ascending']  # 0
                elif is_descending:
                    tone_pattern_id = pattern_ids['descending']  # 1
            
            # Set the range based on the window parameter
            if window == 'default':
                range_value = (start_idx + 1, end_idx)
            elif window == 'pre_odor':
                range_value = (start_idx + 1, end_idx - 2)
            elif window == 'post_odor':
                range_value = (start_idx + 6, end_idx)
            else:
                # Default to the original behavior if an invalid window is specified
                range_value = (start_idx + 1, end_idx)

            result[event_id] = {
                'odor': odor,
                'tone_pattern_id': tone_pattern_id,
                'direction': direction,
                'range': range_value
            }

            event_id += 1
            i = end_idx + 1
        else:
            i += 1

    return result

def test_model(X, y, metric='f1'):
    """
    Test a SVM model with cross-validation.

    Args:
        X: Feature matrix
        y: Target vector
        metric: Evaluation metric ('accuracy', 'f1', or 'roc_auc')

    Returns:
        Mean score across folds
    """
    # Remove classes with fewer than 2 samples
    class_counts = Counter(y)
    valid_classes = {cls for cls, count in class_counts.items() if count >= 2}
    valid_indices = [i for i, label in enumerate(y) if label in valid_classes]

    X = X[valid_indices]
    y = np.array(y)[valid_indices]

    if len(set(y)) < 2:
        raise ValueError("Not enough classes with >=2 samples to perform classification.")

    # Label encode y
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = np.unique(y_encoded)
    n_classes = len(classes)

    # Initialize model
    model = SVC(probability=True, kernel='rbf', random_state=42)

    # Cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in cv.split(X, y_encoded):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if metric == 'accuracy':
            score = accuracy_score(y_test, y_pred)
        elif metric == 'f1':
            score = f1_score(y_test, y_pred, average='macro')
        elif metric == 'roc_auc':
            if n_classes == 2:
                y_score = model.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, y_score)
            else:
                y_score = model.predict_proba(X_test)
                y_test_bin = label_binarize(y_test, classes=classes)
                score = roc_auc_score(y_test_bin, y_score, multi_class='ovr', average='macro')
        else:
            raise ValueError("Unsupported metric. Choose from: 'accuracy', 'f1', 'roc_auc'")

        scores.append(score)

    # print(f"Mean {metric}: {np.mean(scores):.4f}")
    return np.mean(scores)

def test_model_dicts(X_dict, y_dict, metric='f1'):
    """
    Test models on multiple X and y combinations.
    
    Args:
        X_dict: Dictionary of feature matrices
        y_dict: Dictionary of target vectors
        metric: Evaluation metric
        
    Returns:
        Dictionary of results
    """
    results = {}
    for x_key, X_arr in X_dict.items():
        results[x_key] = {}
        for y_key, y_arr in y_dict.items():
            # print(f"Testing model with X: {x_key} and y: {y_key}")
            try:
                # Initialize the nested dictionary structure first
                results[x_key][y_key] = {}
                
                score = test_model(X_arr, y_arr.ravel(), metric=metric)  # flatten y if needed
                results[x_key][y_key]['score'] = score
                results[x_key][y_key]['classes'] = np.unique(y_arr.ravel(), return_counts=True)
            except Exception as e:
                # Handle potential errors gracefully
                print(f"Error processing {x_key}/{y_key}: {str(e)}")
                results[x_key][y_key] = {'score': None, 'error': str(e)}
    return results

def parse_events_data(session_path, probe_list_path="../CODE/behaviour/probe_list.xlsx"):
    """
    Parse events data from a session using two different methods.
    
    Args:
        session_path (str): Path to the session directory
        probe_list_path (str): Path to the probe list Excel file
        
    Returns:
        tuple: (events, event_times) if successful, (None, None) otherwise
    """
    logging.info(f"Attempting to parse events data from {session_path}")
    
    # Try the second method first (using recordingEvents.mat)
    try:
        logging.info("Trying method 2 (recordingEvents.mat)...")
        events_path = os.path.join(session_path, 'recordingEvents.mat')
        
        if not os.path.exists(events_path):
            logging.warning(f"File not found: {events_path}")
            raise FileNotFoundError(f"File not found: {events_path}")
            
        events_data = loadmat(events_path)
        event_idxs = events_data['event']['Strobed'][0][0]
        event_times = events_data['event'][0][0][0]
        
        # Load labels from Excel file
        if not os.path.exists(probe_list_path):
            logging.warning(f"Probe list file not found: {probe_list_path}")
            raise FileNotFoundError(f"Probe list file not found: {probe_list_path}")
            
        labels = pd.read_excel(probe_list_path, header=None)
        labels = labels.to_numpy()
        lookup = dict(labels)
        
        events = np.array([lookup.get(x, 'UNKNOWN') for x in event_idxs.flatten()])
        events = events.reshape(-1, 1)
        
        # Filter out LICKING and _ITI events
        mask = (events.flatten() == 'LICKING') | np.char.endswith(events.flatten(), '_ITI')
        mask = ~mask
        events = list(events[mask].flatten())
        event_times = event_times[mask]
        
        logging.info(f"Method 2 successful: Found {len(events)} events")
        return events, event_times
        
    except Exception as e:
        logging.warning(f"Method 2 failed: {str(e)}")
        logging.info("Trying method 1 (raw_events_pl2.mat)...")
        
        # Try the first method (using raw_events_pl2.mat)
        try:
            events_path = os.path.join(session_path, 'raw_events_pl2.mat')
            
            if not os.path.exists(events_path):
                logging.warning(f"File not found: {events_path}")
                raise FileNotFoundError(f"File not found: {events_path}")
                
            events_data = loadmat(events_path)
            events = events_data['evt'][0][0][2]
            events = [e[0] for e in events.flatten()]
            event_times = events_data['evt'][0][0][0]
            
            logging.info(f"Method 1 successful: Found {len(events)} events")
            return events, event_times
            
        except Exception as e:
            logging.error(f"Both methods failed. Last error: {str(e)}")
            return None, None

def run_analysis(session_path,window):
    """
    Run neural data analysis for a single session.
    
    Args:
        session_path: Path to session directory
        
    Returns:
        Dictionary of model performance scores
    """
    try:
        # Extract mouse and session from path
        path_parts = session_path.split(os.sep)
        mouse = path_parts[-2]
        session = path_parts[-1]
        
        print(f"Analyzing {mouse}/{session}")

        # Load spiking data
        spike_path = os.path.join(session_path, 'extracted_spikes.mat')
        with h5py.File(spike_path, 'r') as f:
            clu = np.array(f['spikeStruct']['clu'])
            ts_sec = np.array(f['spikeStruct']['ts_sec'])

        # Create raster with time bins
        dt = 0.001
        bins = np.arange(0, ts_sec.max() + dt, dt)
        neuron_ids = np.unique(clu)
        neuron_idxs = LabelEncoder().fit_transform(clu.flatten())
        raster = np.zeros((len(neuron_ids), len(bins)-1), dtype=int)
        bin_indices = np.digitize(ts_sec, bins) - 1
        raster[neuron_idxs, bin_indices] += 1

        # Split raster by brain region
        cluster_path = os.path.join(session_path, 'clusterinfo.csv')
        neuron_info = pd.read_csv(cluster_path)
        ofc_mask = neuron_info['region'].str.startswith('OFC')
        hpc_mask = neuron_info['region'].str.startswith('HPC')
        ofc_raster = raster[ofc_mask.to_numpy(), :]
        hpc_raster = raster[hpc_mask.to_numpy(), :]

        # Parse events data
        events,event_times = parse_events_data(session_path, probe_list_path="../CODE/behaviour/probe_list.xlsx")
        parsed_events = extract_odor_events(events,window=window)

        # Initialize data structures
        X_dict = {}
        odor_list = []
        tone_list = []
        direction_list = []
        X_raster = []
        X_hpc = []
        X_ofc = []

        # Process each trial epoch
        for epoch in list(parsed_events.keys()):
            start_idx, end_idx = parsed_events[epoch]['range']
            start_time = event_times[start_idx][0]
            end_time = event_times[end_idx][0]

            # Convert times to bin indices
            start_bin = int(start_time // dt)
            end_bin = int(end_time // dt)

            # Clip to valid range
            start_bin = max(0, start_bin)
            end_bin = min(raster.shape[1], end_bin)

            # Mean firing rates per trial segment
            trial_raster = raster[:, start_bin:end_bin].mean(axis=1)
            X_raster.append(trial_raster)

            if hpc_raster.shape[0] > 0:
                trial_hpc = hpc_raster[:, start_bin:end_bin].mean(axis=1)
                X_hpc.append(trial_hpc)

            if ofc_raster.shape[0] > 0:
                trial_ofc = ofc_raster[:, start_bin:end_bin].mean(axis=1)
                X_ofc.append(trial_ofc)

            # Append label values separately
            odor_list.append(parsed_events[epoch]['odor'])
            tone_list.append(parsed_events[epoch]['tone_pattern_id'])
            direction_list.append(parsed_events[epoch]['direction'])

        # Convert lists to arrays
        odor_arr = np.array(odor_list).reshape(-1, 1)
        tone_arr = np.array(tone_list).reshape(-1, 1)
        direction_arr = np.array(direction_list).reshape(-1, 1)

        # Pack into dict
        y_dict = {
            'odor': odor_arr,
            'tone_pattern_id': tone_arr,
            'direction': direction_arr
        }

        # Convert X to dict based on available data
        X_raster = np.vstack(X_raster)
        has_hpc = len(X_hpc) > 0
        has_ofc = len(X_ofc) > 0

        # Populate X_dict based on available regions
        if has_hpc and has_ofc:
            X_dict['raster'] = X_raster
            X_dict['hpc_raster'] = np.vstack(X_hpc)
            X_dict['ofc_raster'] = np.vstack(X_ofc)
        elif has_hpc and not has_ofc:
            X_dict['hpc_raster'] = X_raster
        elif has_ofc and not has_hpc:
            X_dict['ofc_raster'] = X_raster
        else:
            X_dict['raster'] = X_raster

        # Evaluate models
        scores = test_model_dicts(X_dict, y_dict, metric='f1')
        return scores
        
    except Exception as e:
        logging.error(f"Error in run_analysis for {session_path}: {str(e)}")
        print(f"Analysis failed for {session_path}: {str(e)}")
        return None  # Return None instead of breaking

def main():
    """Main function to process all mice and sessions."""
    # Change to data directory
    os.chdir('../DATA')
    path = os.getcwd()

    # Build the dictionary: dd[mouse][session] = {}
    mice = [d for d in os.listdir(path) if d.startswith('MT') and os.path.isdir(os.path.join(path, d))]
    dd = {mouse: {} for mouse in mice}

    for mouse in mice:
        mouse_dir = os.path.join(path, mouse)
        sessions = [session for session in os.listdir(mouse_dir) 
                   if os.path.isdir(os.path.join(mouse_dir, session))]
        dd[mouse] = {session: {} for session in sessions}

    # Files that must be present
    expected_files = ['extracted_spikes.mat', 'clusterinfo.csv']

    # Iterate over all mice and sessions
    for mouse, sessions in dd.items():
        for session in sessions:
            session_path = os.path.join(path, mouse, session)
            present_files = os.listdir(session_path)

            if all(f in present_files for f in expected_files):
                print(f"[✓] All files found for {mouse} / {session}. Running analysis...")
                try:
                    results = run_analysis(session_path, window='pre_odor')
                    if results is not None:
                        dd[mouse][session] = results
                    # No else needed - if results is None, we simply don't store anything
                except Exception as e:
                    print(f"Error processing {mouse}/{session}: {str(e)}")
                    # Continue to the next session
            else:
                missing = [f for f in expected_files if f not in present_files]
                print(f"[✗] Missing files for {mouse} / {session}: {missing}")

    # Save or display results
    with open('../raiyyan_code/decoding_pre_odor.pkl', 'wb') as f:
        pickle.dump(dd, f)
    print('Results saved to decoding.pkl')

if __name__ == "__main__":
    main()
