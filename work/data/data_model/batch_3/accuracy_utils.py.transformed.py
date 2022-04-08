
import numpy as np
import tensorflow as tf
class StreamingAccuracyStats(object):
  def __init__(self):
    self._how_many_gt = 0
    self._how_many_gt_matched = 0
    self._how_many_fp = 0
    self._how_many_c = 0
    self._how_many_w = 0
    self._gt_occurrence = []
    self._previous_c = 0
    self._previous_w = 0
    self._previous_fp = 0
  def read_ground_truth_file(self, file_name):
    with open(file_name, 'r') as f:
      for line in f:
        line_split = line.strip().split(',')
        if len(line_split) != 2:
          continue
        timestamp = round(float(line_split[1]))
        label = line_split[0]
        self._gt_occurrence.append([label, timestamp])
    self._gt_occurrence = sorted(self._gt_occurrence, key=lambda item: item[1])
  def delta(self):
    fp_delta = self._how_many_fp - self._previous_fp
    w_delta = self._how_many_w - self._previous_w
    c_delta = self._how_many_c - self._previous_c
    if fp_delta == 1:
      recognition_state = '(False Positive)'
    elif c_delta == 1:
      recognition_state = '(Correct)'
    elif w_delta == 1:
      recognition_state = '(Wrong)'
    else:
      raise ValueError('Unexpected state in statistics')
    self._previous_c = self._how_many_c
    self._previous_w = self._how_many_w
    self._previous_fp = self._how_many_fp
    return recognition_state
  def calculate_accuracy_stats(self, found_words, up_to_time_ms,
                               time_tolerance_ms):
    if up_to_time_ms == -1:
      latest_possible_time = np.inf
    else:
      latest_possible_time = up_to_time_ms + time_tolerance_ms
    self._how_many_gt = 0
    for ground_truth in self._gt_occurrence:
      ground_truth_time = ground_truth[1]
      if ground_truth_time > latest_possible_time:
        break
      self._how_many_gt += 1
    self._how_many_fp = 0
    self._how_many_c = 0
    self._how_many_w = 0
    has_gt_matched = []
    for found_word in found_words:
      found_label = found_word[0]
      found_time = found_word[1]
      earliest_time = found_time - time_tolerance_ms
      latest_time = found_time + time_tolerance_ms
      has_matched_been_found = False
      for ground_truth in self._gt_occurrence:
        ground_truth_time = ground_truth[1]
        if (ground_truth_time > latest_time or
            ground_truth_time > latest_possible_time):
          break
        if ground_truth_time < earliest_time:
          continue
        ground_truth_label = ground_truth[0]
        if (ground_truth_label == found_label and
            has_gt_matched.count(ground_truth_time) == 0):
          self._how_many_c += 1
        else:
          self._how_many_w += 1
        has_gt_matched.append(ground_truth_time)
        has_matched_been_found = True
        break
      if not has_matched_been_found:
        self._how_many_fp += 1
    self._how_many_gt_matched = len(has_gt_matched)
  def print_accuracy_stats(self):
    if self._how_many_gt == 0:
      tf.compat.v1.logging.info('No ground truth yet, {}false positives'.format(
          self._how_many_fp))
    else:
      any_match_percentage = self._how_many_gt_matched / self._how_many_gt * 100
      correct_match_percentage = self._how_many_c / self._how_many_gt * 100
      wrong_match_percentage = self._how_many_w / self._how_many_gt * 100
      false_positive_percentage = self._how_many_fp / self._how_many_gt * 100
      tf.compat.v1.logging.info(
          '{:.1f}% matched, {:.1f}% correct, {:.1f}% wrong, '
          '{:.1f}% false positive'.format(any_match_percentage,
                                          correct_match_percentage,
                                          wrong_match_percentage,
                                          false_positive_percentage))
