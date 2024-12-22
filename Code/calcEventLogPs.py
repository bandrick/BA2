from datetime import datetime, timedelta  # Used for timestamp and duration calculations
import os  # Used for file path operations in process_log() for os.path.basename() # Used for importing XES files
import numpy as np
from itertools import combinations
from itertools import chain
from collections import Counter
from collections import Counter
from nltk.metrics.distance import edit_distance


def count_events(log):
    event_count = 0
    for trace in log:
        event_count += len(trace)
    return event_count

def count_distinct_events(log):
    event_types = set()
    for trace in log:
        for event in trace:
            if 'concept:name' in event:
                event_types.add(event['concept:name'])
    return len(event_types)

def count_traces_and_distinct_traces(log):
    distinct_traces = set()
    for trace in log:
        event_names = tuple(event['concept:name'] for event in trace if 'concept:name' in event)
        distinct_traces.add(event_names)
    total_traces = len(log)
    total_distinct_traces = len(distinct_traces)
    return total_traces, total_distinct_traces

def count_distinct_start_end_events(log):
    start_events = set()
    end_events = set()
    for trace in log:
        if len(trace) > 0:
            if 'concept:name' in trace[0]:
                start_events.add(trace[0]['concept:name'])
            if 'concept:name' in trace[-1]:
                end_events.add(trace[-1]['concept:name'])
    return len(start_events), len(end_events)

def trace_length_statistics(log):
    trace_lengths = [len(trace) for trace in log]
    if trace_lengths:
        average_length = sum(trace_lengths) / len(trace_lengths)
        max_length = max(trace_lengths)
        min_length = min(trace_lengths)
    else:
        average_length = max_length = min_length = 0
    return average_length, max_length, min_length

def calculate_ats(log):
    total_distinct_event_classes = sum(len(set(event['concept:name'] for event in trace if 'concept:name' in event)) for trace in log)
    total_traces = len(log)
    ats = total_distinct_event_classes / total_traces if total_traces > 0 else 0
    return ats

def trace_coverage(log):
    trace_dict = {}
    total_traces = len(log)
    for trace in log:
        event_classes = tuple(event['concept:name'] for event in trace if 'concept:name' in event)
        trace_dict[event_classes] = trace_dict.get(event_classes, 0) + 1
    sorted_traces = sorted(trace_dict.items(), key=lambda item: item[1], reverse=True)
    trace_count_80_percent = 0.8 * total_traces
    cumulative_traces = 0
    absolute_trace_coverage = 0
    for _, count in sorted_traces:
        cumulative_traces += count
        absolute_trace_coverage += 1
        if cumulative_traces >= trace_count_80_percent:
            break
    relative_trace_coverage = absolute_trace_coverage / total_traces if total_traces > 0 else 0
    return absolute_trace_coverage, relative_trace_coverage

def event_class(event):
    return event['concept:name']

def event_classes_trace(trace):
    return {event_class(event) for event in trace if 'concept:name' in event}

def event_classes_log(log):
    return set.union(*(event_classes_trace(trace) for trace in log))

def direct_following_relation_trace(trace):
    return {(trace[i]['concept:name'], trace[i+1]['concept:name']) 
            for i in range(len(trace) - 1) 
            if 'concept:name' in trace[i] and 'concept:name' in trace[i+1]}

def direct_following_relation_log(log):
    return set.union(*(direct_following_relation_trace(trace) for trace in log))

def calculate_st_l(log):
    log_following_relations = direct_following_relation_log(log)
    num_following_relations = len(log_following_relations)
    event_classes_set = event_classes_log(log)
    num_event_classes = len(event_classes_set)
    st_l = 1 - num_following_relations / (num_event_classes ** 2) if num_event_classes > 0 else 1
    return st_l

def level_of_detail(log):
    total_distinct_event_classes = sum(len(event_classes_trace(trace)) for trace in log)
    total_traces = len(log)
    lod = total_distinct_event_classes / total_traces if total_traces > 0 else 0
    return lod

# def calculate_affinity(trace1, trace2):
#     F_l = direct_following_relation_trace(trace1)
#     F_m = direct_following_relation_trace(trace2)
#     intersection = F_l.intersection(F_m)
#     union = F_l.union(F_m)
#     affinity = len(intersection) / len(union) if union else 0
#     return affinity

# def mean_affinity(log):
#     n = len(log)
#     if n < 2:
#         return 0
#     sum_affinity = sum(calculate_affinity(log[i], log[j]) for i in range(n) for j in range(i + 1, n))
#     count = n * (n - 1) // 2
#     mean_affinity = sum_affinity / count if count > 0 else 0
    return mean_affinity

def calculate_self_loops_and_sizes(log):
    total_self_loops = 0
    traces_with_self_loops = 0
    total_length_of_self_loops = 0
    total_self_loop_sequences = 0
    for trace in log:
        trace_self_loops = 0
        previous_event_type = None
        current_sequence_length = 0
        for event in trace:
            current_event_type = event['concept:name']
            if current_event_type == previous_event_type:
                current_sequence_length = current_sequence_length + 1 if current_sequence_length > 0 else 2
            else:
                if current_sequence_length > 0:
                    total_length_of_self_loops += current_sequence_length
                    total_self_loop_sequences += 1
                    current_sequence_length = 0
            previous_event_type = current_event_type
            trace_self_loops += current_sequence_length > 0
        if current_sequence_length > 0:
            total_length_of_self_loops += current_sequence_length
            total_self_loop_sequences += 1
        if trace_self_loops > 0:
            traces_with_self_loops += 1
        total_self_loops += trace_self_loops
    average_self_loop_size = (total_length_of_self_loops / total_self_loop_sequences
                              if total_self_loop_sequences > 0 else 0)
    return traces_with_self_loops, total_self_loops, average_self_loop_size

def calculate_duration_throughput_time(log):
    earliest_start = min((trace[0]['time:timestamp'] for trace in log if trace), default=None)
    latest_end = max((trace[-1]['time:timestamp'] for trace in log if trace), default=None)
    if earliest_start is None or latest_end is None:
        return None
    duration = latest_end - earliest_start
    return duration

def time_granularity_trace(trace):
    timestamps = [event['time:timestamp'] for event in trace if 'time:timestamp' in event]
    time_differences = [t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:])]
    return min(time_differences) if time_differences else None

def average_time_granularity(log):
    granularities = [time_granularity_trace(trace) for trace in log]
    valid_granularities = [g for g in granularities if g is not None]
    return sum(valid_granularities, timedelta()) / len(valid_granularities) if valid_granularities else None


# def calculate_mean_affinity(event_log):
#     """
#     Calculate the Mean Affinity (MA) of an event log.

#     Args:
#         event_log (list of list): A list of traces, where each trace is a list of events.

#     Returns:
#         float: The mean affinity of the event log.
#     """
#     def direct_following_relation(trace):
#         """
#         Calculate the set of direct-following relations in a trace.
        
#         Args:
#             trace (list): A single trace (sequence of events).
        
#         Returns:
#             set: A set of tuples representing direct-following relations.
#         """
#         return set((trace[i], trace[i + 1]) for i in range(len(trace) - 1))

#     def affinity(trace1, trace2):
#         """
#         Calculate the affinity between two traces.
        
#         Args:
#             trace1 (list): The first trace (sequence of events).
#             trace2 (list): The second trace (sequence of events).
        
#         Returns:
#             float: The affinity value between the two traces.
#         """
#         F1 = direct_following_relation(trace1)
#         F2 = direct_following_relation(trace2)
#         if not F1 and not F2:  # Handle empty direct-following relations
#             return 0
#         return len(F1 & F2) / len(F1 | F2)


#     trace_pairs = combinations(event_log, 2)
#     num_traces = len(event_log)

#     total_affinity = sum(affinity(trace1, trace2) for trace1, trace2 in trace_pairs)

#     denominator = num_traces * (num_traces - 1)  # |L| * (|L| - 1)
#     return total_affinity / denominator if denominator != 0 else 0

def calculate_event_diversity(event_log):
    """
    Calculate the Event Diversity (D) of an event log.

    Args:
        event_log (list of list): A list of traces, where each trace is a list of events.

    Returns:
        float: The event diversity of the log.
    """

    unique_event_combinations = {frozenset(trace) for trace in event_log}


    return len(unique_event_combinations) / len(event_log) if event_log else 0

def calculate_event_repeatability(event_log):
    """
    Calculate the Event Repeatability (ER) of an event log.

    Args:
        event_log (list of list): A list of traces, where each trace is a list of events.

    Returns:
        float: The event repeatability of the log.
    """
    total_repeats = 0
    total_events = 0

    for trace in event_log:
        event_counts = {event: trace.count(event) for event in set(trace)}
        repeats = sum(count - 1 for count in event_counts.values() if count > 1)
        total_repeats += repeats
        total_events += len(set(trace))

    return total_repeats / total_events if total_events > 0 else 0


def calculate_transition_consistency(event_log):
    """
    Calculate the Transition Consistency (TC) of an event log.

    Args:
        event_log (list of list): A list of traces, where each trace is a list of events.

    Returns:
        float: The transition consistency of the log.
    """
    def direct_following_relation(trace):
        return set((trace[i], trace[i + 1]) for i in range(len(trace) - 1))

 
    trace_relations = [direct_following_relation(trace) for trace in event_log]


    global_relations = set(chain.from_iterable(trace_relations))


    avg_relations_per_trace = sum(len(relations) for relations in trace_relations) / len(event_log)

  
    return 1 - (len(global_relations) / (len(event_log) * avg_relations_per_trace)) if avg_relations_per_trace > 0 else 0

def calculate_sequential_complexity(event_log):
    """
    Calculate the Sequential Complexity (SC) of an event log.

    Args:
        event_log (list of list): A list of traces, where each trace is a list of events.

    Returns:
        float: The sequential complexity of the log.
    """
    total_transitions = 0
    total_events = 0

    for trace in event_log:
        total_transitions += len(trace) - 1  # Transitions are len(trace) - 1
        total_events += len(trace)

    return total_transitions / total_events if total_events > 0 else 0



# def calculate_cross_trace_similarity(event_log):
#     """
#     Calculate the Cross-Trace Similarity (CTS) of an event log.

#     Args:
#         event_log (list of list): A list of traces, where each trace is a list of events.

#     Returns:
#         float: The cross-trace similarity of the log.
#     """
#     def jaccard_similarity(set1, set2):
#         return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

#     # Extract event sets for all traces
#     trace_event_sets = [set(trace) for trace in event_log]

#     # Calculate pairwise Jaccard similarity
#     similarities = [
#         jaccard_similarity(trace_event_sets[i], trace_event_sets[j])
#         for i, j in combinations(range(len(trace_event_sets)), 2)
#     ]

#     # Mean similarity across all pairs
#     return sum(similarities) / len(similarities) if similarities else 0

def calculate_rare_sequence_impact(event_log, percentage=0.05):
    """
    Calculate the Rare Sequence Impact (RSI) of an event log.

    Args:
        event_log (list of list): A list of traces, where each trace is a list of events.
        percentage (float): The percentage (0-1) to define the threshold for rare sequences.
                          Default is 0.01 (1% of total traces)

    Returns:
        float: The rare sequence impact of the log.
    """
 
    threshold = max(1, round(len(event_log) * percentage))
    if(len(event_log) == 1):
        return 0

    subsequences = Counter()
    for trace in event_log:
        for length in range(1, len(trace) + 1):
            for i in range(len(trace) - length + 1):
                subsequences[tuple(trace[i:i + length])] += 1


    rare_sum = sum(len(subseq) for subseq, count in subsequences.items() if count <= threshold)


    total_length = sum(len(trace) for trace in event_log)

    return rare_sum / total_length if total_length > 0 else 0


def calculate_event_class_dispersion(event_log):
    """
    Calculate the Event Class Dispersion (ECD) of an event log.

    Args:
        event_log (list of list): A list of traces, where each trace is a list of events.

    Returns:
        float: The event class dispersion of the log.
    """


    event_counts = Counter(event for trace in event_log for event in trace)

    return np.std(list(event_counts.values()))

def calculate_event_cooccurrence_consistency(event_log):
    """
    Calculate the Event Co-occurrence Consistency (ECC) of an event log.

    Args:
        event_log (list of list): A list of traces, where each trace is a list of events.

    Returns:
        float: The event co-occurrence consistency of the log.
    """


    unique_events = set(event for trace in event_log for event in trace)

    event_counts = Counter(event for trace in event_log for event in trace)

    cooccurrence_counts = Counter()
    for trace in event_log:
        event_set = set(trace)
        for e1, e2 in combinations(event_set, 2):
            cooccurrence_counts[frozenset([e1, e2])] += 1

    numerator = 0
    for event_pair, freq in cooccurrence_counts.items():
        e1, e2 = event_pair
        numerator += freq / (event_counts[e1] * event_counts[e2])

    num_pairs = len(unique_events) * (len(unique_events) - 1) / 2
    return numerator / num_pairs if num_pairs > 0 else 0

def calculate_trace_variability(event_log):
    
  

    trace_counts = Counter(tuple(trace) for trace in event_log)
    template_trace = max(trace_counts, key=trace_counts.get)

    total_distance = sum(edit_distance(template_trace, trace) for trace in event_log)
    return total_distance / len(event_log) if event_log else 0

def calculate_event_log_ps(log):
    
    results = {
        #'File': os.path.basename(file_path),
        'Number of Events': count_events(log),
        'ATS': calculate_ats(log),
        'Number of Traces': count_traces_and_distinct_traces(log)[0],
        'Distinct Events': count_distinct_events(log),
        'Distinct Traces': count_traces_and_distinct_traces(log)[1],
        'Distinct Start Events': count_distinct_start_end_events(log)[0],
        'Distinct End Events': count_distinct_start_end_events(log)[1]
    }
    #print(1)
    avgT, maxT, minT = trace_length_statistics(log)
    results.update({
        'Average Trace Length': avgT,
        'Max Trace Length': maxT,
        'Min Trace Length': minT,
        'Event Density': results['ATS'] / avgT if avgT != 0 else 0
    })
    #print(2)
    absolute_coverage, relative_coverage = trace_coverage(log)
    results.update({
        'Absolute Trace Coverage': absolute_coverage,
        'Relative Trace Coverage': relative_coverage,
        'Structure': calculate_st_l(log),
        #Mean Affinity': mean_affinity(log),
        'Level of Detail': level_of_detail(log)
    })
    #print(3)
    traces_with_self_loops, total_self_loops, average_self_loop_size = calculate_self_loops_and_sizes(log)
    results.update({
        'Traces with Self-loops': traces_with_self_loops,
        'Total Self-loops': total_self_loops,
        'Average Self-loop Size': average_self_loop_size
    })
    #print(4)
    results.update({
        #'Mean Affinity': calculate_mean_affinity(log),
        'Event Diversity': calculate_event_diversity(log),
        'Event Repeatability': calculate_event_repeatability(log),
    })
    #print(5)
    results.update({
        'Transition Consistency': calculate_transition_consistency(log),
        'Sequential Complexity': calculate_sequential_complexity(log),
        #'Cross-Trace Similarity': calculate_cross_trace_similarity(log),
        'Rare Sequence Impact': calculate_rare_sequence_impact(log),
        'Event Class Dispersion': calculate_event_class_dispersion(log),
        'Event Co-occurrence Consistency': calculate_event_cooccurrence_consistency(log),
        'Trace Variability': calculate_trace_variability(log)
    })
    #print(6)
    return results


#log = xes_importer.apply('/Users/benjaminandrick/Documents/Studium/Semester 7/Bachelorarbeit/Code/Logs/Hospital_log.xes')
#process_log(log)



# def count_events(log):
#     event_count = 0
#     for trace in log:
#         event_count += len(trace)
#     return event_count

# def count_distinct_events(log):
#     event_types = set()
#     for trace in log:
#         for event in trace:
#             if 'concept:name' in event:
#                 event_types.add(event['concept:name'])
#     return len(event_types)

# def count_traces_and_distinct_traces(log):
#     distinct_traces = set()
#     for trace in log:
#         event_names = tuple(event['concept:name'] for event in trace if 'concept:name' in event)
#         distinct_traces.add(event_names)
#     total_traces = len(log)
#     total_distinct_traces = len(distinct_traces)
#     return total_traces, total_distinct_traces

# def count_distinct_start_end_events(log):
#     start_events = set()
#     end_events = set()
#     for trace in log:
#         if len(trace) > 0:
#             if 'concept:name' in trace[0]:
#                 start_events.add(trace[0]['concept:name'])
#             if 'concept:name' in trace[-1]:
#                 end_events.add(trace[-1]['concept:name'])
#     return len(start_events), len(end_events)

# def trace_length_statistics(log):
#     trace_lengths = [len(trace) for trace in log]
#     if trace_lengths:
#         average_length = sum(trace_lengths) / len(trace_lengths)
#         max_length = max(trace_lengths)
#         min_length = min(trace_lengths)
#     else:
#         average_length = max_length = min_length = 0
#     return average_length, max_length, min_length

# def calculate_ats(log):
#     total_distinct_event_classes = sum(len(set(event['concept:name'] for event in trace if 'concept:name' in event)) for trace in log)
#     total_traces = len(log)
#     ats = total_distinct_event_classes / total_traces if total_traces > 0 else 0
#     return ats

# def trace_coverage(log):
#     trace_dict = {}
#     total_traces = len(log)
#     for trace in log:
#         event_classes = tuple(event['concept:name'] for event in trace if 'concept:name' in event)
#         trace_dict[event_classes] = trace_dict.get(event_classes, 0) + 1
#     sorted_traces = sorted(trace_dict.items(), key=lambda item: item[1], reverse=True)
#     trace_count_80_percent = 0.8 * total_traces
#     cumulative_traces = 0
#     absolute_trace_coverage = 0
#     for _, count in sorted_traces:
#         cumulative_traces += count
#         absolute_trace_coverage += 1
#         if cumulative_traces >= trace_count_80_percent:
#             break
#     relative_trace_coverage = absolute_trace_coverage / total_traces if total_traces > 0 else 0
#     return absolute_trace_coverage, relative_trace_coverage

# def event_class(event):
#     return event['concept:name']

# def event_classes_trace(trace):
#     return {event_class(event) for event in trace if 'concept:name' in event}

# def event_classes_log(log):
#     return set.union(*(event_classes_trace(trace) for trace in log))

# def direct_following_relation_trace(trace):
#     return {(trace[i]['concept:name'], trace[i+1]['concept:name']) 
#             for i in range(len(trace) - 1) 
#             if 'concept:name' in trace[i] and 'concept:name' in trace[i+1]}

# def direct_following_relation_log(log):
#     return set.union(*(direct_following_relation_trace(trace) for trace in log))

# def calculate_st_l(log):
#     log_following_relations = direct_following_relation_log(log)
#     num_following_relations = len(log_following_relations)
#     event_classes_set = event_classes_log(log)
#     num_event_classes = len(event_classes_set)
#     st_l = 1 - num_following_relations / (num_event_classes ** 2) if num_event_classes > 0 else 1
#     return st_l

# def level_of_detail(log):
#     total_distinct_event_classes = sum(len(event_classes_trace(trace)) for trace in log)
#     total_traces = len(log)
#     lod = total_distinct_event_classes / total_traces if total_traces > 0 else 0
#     return lod

# def calculate_affinity(trace1, trace2):
#     F_l = direct_following_relation_trace(trace1)
#     F_m = direct_following_relation_trace(trace2)
#     intersection = F_l.intersection(F_m)
#     union = F_l.union(F_m)
#     affinity = len(intersection) / len(union) if union else 0
#     return affinity

# def mean_affinity(log):
#     n = len(log)
#     if n < 2:
#         return 0
#     sum_affinity = sum(calculate_affinity(log[i], log[j]) for i in range(n) for j in range(i + 1, n))
#     count = n * (n - 1) // 2
#     mean_affinity = sum_affinity / count if count > 0 else 0
#     return mean_affinity

# def calculate_self_loops_and_sizes(log):
#     total_self_loops = 0
#     traces_with_self_loops = 0
#     total_length_of_self_loops = 0
#     total_self_loop_sequences = 0
#     for trace in log:
#         trace_self_loops = 0
#         previous_event_type = None
#         current_sequence_length = 0
#         for event in trace:
#             current_event_type = event['concept:name']
#             if current_event_type == previous_event_type:
#                 current_sequence_length = current_sequence_length + 1 if current_sequence_length > 0 else 2
#             else:
#                 if current_sequence_length > 0:
#                     total_length_of_self_loops += current_sequence_length
#                     total_self_loop_sequences += 1
#                     current_sequence_length = 0
#             previous_event_type = current_event_type
#             trace_self_loops += current_sequence_length > 0
#         if current_sequence_length > 0:
#             total_length_of_self_loops += current_sequence_length
#             total_self_loop_sequences += 1
#         if trace_self_loops > 0:
#             traces_with_self_loops += 1
#         total_self_loops += trace_self_loops
#     average_self_loop_size = (total_length_of_self_loops / total_self_loop_sequences
#                               if total_self_loop_sequences > 0 else 0)
#     return traces_with_self_loops, total_self_loops, average_self_loop_size

# def calculate_duration_throughput_time(log):
#     earliest_start = min((trace[0]['time:timestamp'] for trace in log if trace), default=None)
#     latest_end = max((trace[-1]['time:timestamp'] for trace in log if trace), default=None)
#     if earliest_start is None or latest_end is None:
#         return None
#     duration = latest_end - earliest_start
#     return duration

# def time_granularity_trace(trace):
#     timestamps = [event['time:timestamp'] for event in trace if 'time:timestamp' in event]
#     time_differences = [t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:])]
#     return min(time_differences) if time_differences else None

# def average_time_granularity(log):
#     granularities = [time_granularity_trace(trace) for trace in log]
#     valid_granularities = [g for g in granularities if g is not None]
#     return sum(valid_granularities, timedelta()) / len(valid_granularities) if valid_granularities else None

# def calculate_event_log_ps(log):
    
#     results = {
#         #'File': os.path.basename(file_path),
#         'Number of Events': count_events(log),
#         'ATS': calculate_ats(log),
#         'Number of Traces': count_traces_and_distinct_traces(log)[0],
#         'Distinct Events': count_distinct_events(log),
#         'Distinct Traces': count_traces_and_distinct_traces(log)[1],
#         'Distinct Start Events': count_distinct_start_end_events(log)[0],
#         'Distinct End Events': count_distinct_start_end_events(log)[1]
#     }
#     avgT, maxT, minT = trace_length_statistics(log)
#     results.update({
#         'Average Trace Length': avgT,
#         'Max Trace Length': maxT,
#         'Min Trace Length': minT,
#         'Event Density': results['ATS'] / avgT if avgT != 0 else 0
#     })
#     absolute_coverage, relative_coverage = trace_coverage(log)
#     results.update({
#         'Absolute Trace Coverage': absolute_coverage,
#         'Relative Trace Coverage': relative_coverage,
#         'Structure': calculate_st_l(log),
#         #'Mean Affinity': mean_affinity(log),
#         'Level of Detail': level_of_detail(log)
#     })
#     traces_with_self_loops, total_self_loops, average_self_loop_size = calculate_self_loops_and_sizes(log)
#     results.update({
#         'Traces with Self-loops': traces_with_self_loops,
#         'Total Self-loops': total_self_loops,
#         'Average Self-loop Size': average_self_loop_size
#     })

#     return results


#log = xes_importer.apply('/Users/benjaminandrick/Documents/Studium/Semester 7/Bachelorarbeit/Code/Logs/Hospital_log.xes')
#process_log(log)