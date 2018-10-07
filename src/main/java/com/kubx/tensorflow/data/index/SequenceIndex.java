package com.kubx.tensorflow.data.index;

import com.kubx.tensorflow.data.Index;

public class SequenceIndex extends Index {

  @Override
  public int size() {
    return sequence.length;
  }

  @Override
  public int position(int i) {
    return sequence[i];
  }
  
  public SequenceIndex(int... sequence) {
    this.sequence = sequence;
  }
  
  private int[] sequence;
}
