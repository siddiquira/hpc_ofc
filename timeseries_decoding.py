import os
import h5py
import numpy as np
import pandas as pd
import re
import pickle
import logging
from collections import Counter
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
import argparse
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
    
    # Try method 2 first (using recordingEvents.mat)
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
        
        # Try method 1 (using raw_events_pl2.mat)
        try:
            logging.info("Trying method 1 (raw_events_pl2.mat)...")
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

def parse_beh(directory_path, session=None):
    """
    Parses .beh files depending on session type.

    Args:
        directory_path (str): Path to directory containing the .beh file(s).
        session (str, optional): Session name to determine parsing logic.

    Returns:
        dict: {'left': [...], 'right': [...]} with non -1 values, no duplicates.
    """

    def parse_single_beh_file(path):
        # (Same code to parse a single .beh file)
        beh_files = [f for f in os.listdir(path) if f.endswith('.beh')]
        if not beh_files:
            raise FileNotFoundError(f"No .beh file found in {path}")
        if len(beh_files) > 1:
            raise RuntimeError(f"Expected one .beh file, found multiple in {path}")
        file_path = os.path.join(path, beh_files[0])

        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        data = {'left': [], 'right': []}
        for i in range(13, 29, 2):
            label = lines[i].lower()
            try:
                value = int(lines[i + 1])
            except (ValueError, IndexError):
                continue
            if value == -1:
                continue
            if 'left' in label:
                data['left'].append(value)
            elif 'right' in label:
                data['right'].append(value)
        return data

    # Initialize results dict
    combined_data = {'left': [], 'right': []}

    if session and session.startswith("VC_Set"):
        # Parse /set1 and /set2 subdirectories, then merge
        for subset in ['set1', 'set2']:
            sub_path = os.path.join(directory_path, subset)
            if not os.path.isdir(sub_path):
                raise FileNotFoundError(f"Expected directory not found: {sub_path}")
            data = parse_single_beh_file(sub_path)
            # Append non-duplicate values only
            for side in ['left', 'right']:
                for val in data[side]:
                    if val not in combined_data[side]:
                        combined_data[side].append(val)

    elif session and session.startswith("VC_Stable"):
        # Parse normally in directory_path
        combined_data = parse_single_beh_file(directory_path)

    else:
        # Default fallback: parse normally
        combined_data = parse_single_beh_file(directory_path)

    return combined_data

def assign_odor_directions(result_dict, forced_choice_dict):
    for event_id, info in result_dict.items():
        odor_str = info['odor']
        # Extract the odor number as int, assuming format 'Odor_XX'
        try:
            odor_num = int(odor_str.split('_')[1])
        except (IndexError, ValueError):
            # If parsing fails, skip or set to None
            info['odor_direction'] = None
            continue
        
        # Decide left or right
        if odor_num in forced_choice_dict['left']:
            info['odor_direction'] = 'left'
        elif odor_num in forced_choice_dict['right']:
            info['odor_direction'] = 'right'
        else:
            info['odor_direction'] = None  # odor number not found in either list

    return result_dict

def extract_odor_events(session_path,session,event_array):
    """
    Extract odor events from event array with pattern identification.
    
    Args:
        event_array: List of event strings
        
    Returns:
        Dictionary of events and tone patterns (ascending=0, descending=1)
    """
    result = {}
    pattern_ids = {
        'ascending': 0,  # Ascending pattern gets ID 0
        'descending': 1  # Descending pattern gets ID 1
    }
    event_id = 0

    # Regex patterns for event identification
    odor_pattern = re.compile(r'^Odor_(\d+)$')
    tone_on_pattern = re.compile(r'^TONE_ON_(\d+)Hz$')

    i = 0
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

            # Extract tone frequencies and determine pattern
            tone_frequencies = []
            for j in range(start_idx, end_idx + 1):
                match = tone_on_pattern.match(event_array[j])
                if match:
                    freq = int(match.group(1))
                    tone_frequencies.append(freq)
            
            # Determine pattern type (ascending or descending)
            tone_pattern_id = None
            if tone_frequencies:
                is_ascending = all(tone_frequencies[i] <= tone_frequencies[i+1] 
                                  for i in range(len(tone_frequencies)-1))
                is_descending = all(tone_frequencies[i] >= tone_frequencies[i+1] 
                                   for i in range(len(tone_frequencies)-1))
                
                if is_ascending:
                    tone_pattern_id = pattern_ids['ascending']
                elif is_descending:
                    tone_pattern_id = pattern_ids['descending']
            
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

    odor_directions = parse_beh(directory_path=f'{session_path}/Behaviour',session=session)
    result = assign_odor_directions(result,odor_directions)

    return result

def test_model(X, y):
    """
    Test a SVM model with cross-validation.

    Args:
        X: Feature matrix
        y: Target vector

    Returns:
        Mean score across folds
    """
    # Remove classes with fewer than 10 samples
    class_counts = Counter(y)
    valid_classes = {cls for cls, count in class_counts.items() if count >= 10}
    valid_indices = [i for i, label in enumerate(y) if label in valid_classes]

    X = X[valid_indices]
    y = np.array(y)[valid_indices]

    # if len(set(y)) < 10:
    #     raise ValueError("Not enough classes with >=10 samples to perform classification.")

    # Label encode y
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    # classes = np.unique(y_encoded)
    # n_classes = len(classes)

    # Collect F1 score across 100 draws
    scores = []
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=.2, stratify=y_encoded)
        model = SVC(probability=True, kernel='rbf')
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        scores.append(f1_score(y_test,y_pred,average='weighted'))

    return np.mean(scores),np.std(scores)/np.sqrt(100)

def test_model_dicts(X_dict, y_dict):
    """
    Test models on multiple X and y combinations.
    
    Args:
        X_dict: Dictionary of feature matrices
        y_dict: Dictionary of target vectors
        
    Returns:
        Dictionary of results
    """
    results = {}
    for x_key, X_arr in X_dict.items():
        results[x_key] = {}
        results[x_key]['n_neurons'] = X_arr.shape[1]
        for y_key, y_arr in y_dict.items():
            logging.debug(f"Testing model with X: {x_key} and y: {y_key}")
            try:
                # Initialize nested dictionary structure
                results[x_key][y_key] = {}
                
                # Test the model and store results
                score = test_model(X_arr, y_arr.ravel())
                results[x_key][y_key]['score'] = score
                # results[x_key][y_key]['classes'] = np.unique(y_arr.ravel(), return_counts=True)
            except Exception as e:
                # Handle errors gracefully
                logging.error(f"Error processing {x_key}/{y_key}: {str(e)}")
                results[x_key][y_key] = {'score': None, 'error': str(e)}
    return results

def run_analysis(session_path, session, window_size_ms=50, min_percent_samples=.9):
    """
    Run decoding analysis for a single session with equal windows per trial.
    Now includes 500ms pre-trial and 500ms post-trial analysis.
    
    Args:
        session_path: Path to session directory
        session: Session identifier
        window_size_ms: Size of time windows in milliseconds (default 50ms)
        min_samples_per_window: Minimum number of samples required per window (default 50)
        
    Returns:
        Dictionary of model performance scores
    """
    try:
        # Extract mouse and session from path
        path_parts = session_path.split(os.sep)
        mouse = path_parts[-2]
        session = path_parts[-1]
        
        logging.info(f"Analyzing {mouse}/{session} with {window_size_ms}ms windows")

        # Load spiking data
        spike_path = os.path.join(session_path, 'extracted_spikes.mat')
        with h5py.File(spike_path, 'r') as f:
            clu = np.array(f['spikeStruct']['clu'])
            ts_sec = np.array(f['spikeStruct']['ts_sec'])

        # Create raster with time bins
        dt = 0.001  # 1ms bins
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
        events, event_times = parse_events_data(session_path)
        if events is None or event_times is None:
            raise ValueError("Failed to parse events data")
            
        parsed_events = extract_odor_events(session_path, session, events) 
        total_samples = len(list(parsed_events.keys()))
        min_samples_per_window = int(total_samples*min_percent_samples)

        # Convert window size to bins
        window_size_bins = int(window_size_ms / 1000 / dt)
        pre_post_duration_bins = int(0.5 / dt)  # 500ms in bins
        
        logging.info(f"Using {window_size_ms}ms ({window_size_bins} bins) windows per trial")
        logging.info(f"Including 500ms pre/post trial periods ({pre_post_duration_bins} bins each)")
        
        # Initialize window-based data structures
        window_data = {}

        # Process each trial epoch (including pre-trial and post-trial periods)
        max_windows = 0
        
        for epoch in parsed_events.keys():
            start_idx, end_idx = parsed_events[epoch]['range']
            start_time = event_times[start_idx][0]
            end_time = event_times[end_idx][0]

            # Convert times to bin indices
            start_bin = max(0, int(start_time // dt))
            end_bin = min(raster.shape[1], int(end_time // dt))
            
            # Extend the analysis period to include 500ms before and after
            extended_start_bin = max(0, start_bin - pre_post_duration_bins)
            extended_end_bin = min(raster.shape[1], end_bin + pre_post_duration_bins)
            
            # Calculate extended trial duration
            extended_duration_bins = extended_end_bin - extended_start_bin
            
            # Skip trials that are too short for even one window
            if extended_duration_bins < window_size_bins:
                logging.warning(f"Extended trial {epoch} too short ({extended_duration_bins} bins) for window size ({window_size_bins} bins)")
                continue
            
            # Fit as many non-overlapping windows as possible in this extended trial
            num_windows_this_trial = extended_duration_bins // window_size_bins
            max_windows = max(max_windows, num_windows_this_trial)
            
            for w in range(num_windows_this_trial):
                # Initialize this window position if we haven't seen it yet
                if w not in window_data:
                    window_data[w] = {
                        'raster': [],
                        'hpc_raster': [],
                        'ofc_raster': [],
                        'odor': [],
                        'odor_direction': [],
                        'tone_pattern_id': [],
                        'direction': []
                    }
                
                window_start = extended_start_bin + w * window_size_bins
                window_end = window_start + window_size_bins
                
                # Extract firing rates for this window
                window_raster = raster[:, window_start:window_end].mean(axis=1)
                window_data[w]['raster'].append(window_raster)

                if hpc_raster.shape[0] > 0:
                    window_hpc = hpc_raster[:, window_start:window_end].mean(axis=1)
                    window_data[w]['hpc_raster'].append(window_hpc)

                if ofc_raster.shape[0] > 0:
                    window_ofc = ofc_raster[:, window_start:window_end].mean(axis=1)
                    window_data[w]['ofc_raster'].append(window_ofc)

                # Append label values for each window (same labels for pre/trial/post)
                window_data[w]['odor'].append(parsed_events[epoch]['odor'])
                window_data[w]['odor_direction'].append(parsed_events[epoch]['odor_direction'])
                window_data[w]['tone_pattern_id'].append(parsed_events[epoch]['tone_pattern_id'])
                window_data[w]['direction'].append(parsed_events[epoch]['direction'])

        if len(window_data) == 0:
            logging.warning(f"No valid windows found for session {session}")
            return None

        # Now organize data by window position and create X_dict, y_dict for each
        X_dict = {}
        total_windows = 0
        valid_windows = 0
        
        has_hpc = hpc_raster.shape[0] > 0
        has_ofc = ofc_raster.shape[0] > 0
        
        for w in range(max_windows):
            if w not in window_data:
                continue
            
            # Check if this window has enough samples
            num_samples = len(window_data[w]['odor'])
            if num_samples < min_samples_per_window:
                logging.info(f"Skipping window_{w}: only {num_samples} samples (< {min_samples_per_window} required)")
                continue
                
            # Convert lists to arrays for this window position
            odor_arr = np.array(window_data[w]['odor']).reshape(-1, 1)
            odor_direction_arr = np.array(window_data[w]['odor_direction']).reshape(-1, 1)
            tone_arr = np.array(window_data[w]['tone_pattern_id']).reshape(-1, 1)
            direction_arr = np.array(window_data[w]['direction']).reshape(-1, 1)

            # Pack labels for this window
            y_dict_w = {
                'odor': odor_arr,
                'odor_direction': odor_direction_arr,
                'tone_pattern_id': tone_arr,
                'direction': direction_arr
            }

            # Convert X data for this window
            X_raster_w = np.vstack(window_data[w]['raster'])
            window_X_dict = {}
            
            if has_hpc and has_ofc:
                window_X_dict['raster'] = X_raster_w
                window_X_dict['hpc_raster'] = np.vstack(window_data[w]['hpc_raster'])
                window_X_dict['ofc_raster'] = np.vstack(window_data[w]['ofc_raster'])
            elif has_hpc and not has_ofc:
                window_X_dict['hpc_raster'] = X_raster_w
            elif has_ofc and not has_hpc:
                window_X_dict['ofc_raster'] = X_raster_w
            else:
                window_X_dict['raster'] = X_raster_w
            
            X_dict[f'window_{w}'] = {'X': window_X_dict, 'y': y_dict_w}
            total_windows += len(X_raster_w)
            valid_windows += 1

        logging.info(f"Created {total_windows} total windows across {valid_windows} valid temporal positions (of {max_windows} possible) from {len(parsed_events)} trials")
        logging.info(f"Each trial extended by 1000ms total (500ms pre + 500ms post)")

        # Evaluate models for each valid window position
        scores = {}
        for w in range(max_windows):
            if f'window_{w}' not in X_dict:
                continue
            window_scores = test_model_dicts(X_dict[f'window_{w}']['X'], X_dict[f'window_{w}']['y'])
            scores[f'window_{w}'] = window_scores
            
        scores['max_windows'] = max_windows
        scores['valid_windows'] = valid_windows
        scores['total_windows'] = total_windows
        scores['window_size_ms'] = window_size_ms
        scores['min_samples_per_window'] = min_samples_per_window
        scores['pre_post_duration_ms'] = 500
        
        return scores
        
    except Exception as e:
        logging.error(f"Error in run_analysis for {session_path}: {str(e)}")
        return None

def main():
    """Main function to process a specific mouse and session."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process neural data with decoding done over period of time')
    parser.add_argument('--mouse', type=str, help='Mouse identifier (e.g., MT001)')
    parser.add_argument('--session', type=str, help='Session identifier')
    args = parser.parse_args()
    
    print(f"Arguments parsed: mouse={args.mouse}, session={args.session}")
    sys.stdout.flush()

    # Set up paths
    data_dir = '../DATA'
    output_path = f'../raiyyan_code/timeseries_decoding_results/{args.mouse}_{args.session}.pkl'
    expected_files = ['extracted_spikes.mat', 'clusterinfo.csv']
    
    original_dir = os.getcwd()
    os.chdir(data_dir)
    path = os.getcwd()

    # Build session path
    session_path = os.path.join(path, args.mouse, args.session)
    
    # Check if session directory exists
    if not os.path.exists(session_path):
        logging.error(f"Session path does not exist: {session_path}")
        os.chdir(original_dir)
        return

    present_files = os.listdir(session_path)

    # Check if all required files are present
    if all(f in present_files for f in expected_files):
        logging.info(f"[✓] All files found for {args.mouse} / {args.session}. Running analysis...")
        try:
            analysis_results = run_analysis(session_path, args.session)
            
            if analysis_results is not None:
                # Create nested dictionary structure: dd[mouse][session] = analysis_results
                results = {args.mouse: {args.session: analysis_results}}
                
                # Save results
                os.chdir(original_dir)
                with open(output_path, 'wb') as f:
                    pickle.dump(results, f)
                logging.info(f'Results saved to {output_path}')
            else:
                logging.error(f"Analysis returned None for {args.mouse}/{args.session}")
        except Exception as e:
            logging.error(f"Error processing {args.mouse}/{args.session}: {str(e)}")
    else:
        missing = [f for f in expected_files if f not in present_files]
        logging.warning(f"[✗] Missing files for {args.mouse} / {args.session}: {missing}")
    
    os.chdir(original_dir)

if __name__ == "__main__":
    main()