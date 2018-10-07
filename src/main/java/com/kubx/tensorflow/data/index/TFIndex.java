package com.kubx.tensorflow.data.index;

public final class TFIndex {
  
  public static final FullIndex FULL = new FullIndex();
  
  public static RangeIndex range(int start, int end) {
    return new RangeIndex(start, end);
  }
  
  public static SequenceIndex sequence(int... sequence) {
    return new SequenceIndex(sequence);
  }
  
  private TFIndex() {}
}
