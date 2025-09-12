"""
Text processing module for video transcription.
Handles text manipulation, sentence splitting, and cue merging.
"""

import re


def extract_words(text):
    """
    Extract words from text using regex.
    
    Args:
        text (str): Input text
    
    Returns:
        set: Set of unique words
    """
    return set(re.findall(r'\b[\w\']+\b', text.lower()))


def split_at_sentence_end(text, word_data):
    """
    Split text at sentence boundaries and align with word timing data.
    
    Args:
        text (str): Input text
        word_data (list): List of word timing data
    
    Returns:
        list: List of sentence segments with timing
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    current_word_index = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            sentence_word_count = len(sentence.split())
            sentence_word_data = word_data[current_word_index:current_word_index + sentence_word_count]
            
            if sentence_word_data:
                start_time = next((word['start'] for word in sentence_word_data if 'start' in word), None)
                end_time = next((word['end'] for word in reversed(sentence_word_data) if 'end' in word), None)
                
                if start_time is not None and end_time is not None:
                    result.append({
                        'text': sentence,
                        'start': start_time,
                        'end': end_time
                    })
                else:
                    # If start or end time is missing, use the previous valid timestamp
                    if result:
                        prev_end = result[-1]['end']
                        result.append({
                            'text': sentence,
                            'start': prev_end,
                            'end': prev_end + 1  # Add 1 second as a placeholder duration
                        })
                    else:
                        # If it's the first sentence and times are missing, use 0 as start time
                        result.append({
                            'text': sentence,
                            'start': 0,
                            'end': 1  # Add 1 second as a placeholder duration
                        })
            current_word_index += sentence_word_count
    
    return result


def merge_short_cues(cues, min_duration=3):
    """
    Merge subtitle cues that are too short.
    
    Args:
        cues (list): List of subtitle cues
        min_duration (float): Minimum duration in seconds
    
    Returns:
        list: List of merged cues
    """
    merged_cues = []
    current_cue = None

    for cue in cues:
        if current_cue is None:
            current_cue = cue
        else:
            duration = cue['end'] - current_cue['start']
            if duration < min_duration:
                current_cue['text'] += ' ' + cue['text']
                current_cue['end'] = cue['end']
            else:
                merged_cues.append(current_cue)
                current_cue = cue

    if current_cue:
        merged_cues.append(current_cue)

    return merged_cues
