package com.kubx.tensorflow.data.index;

import com.kubx.tensorflow.data.Index;

public class RangeIndex extends Index {
  
  @Override
  public int size() {
    return end - start;
  }

  @Override
  public int position(int index) {
    return start + index;
  }
  
  public RangeIndex(int end) {
    this.start = 0;
    this.end = end;
  }

  public RangeIndex(int start, int end) {
    this.start = start;
    this.end = end;
  }
  
  private final int start;
  private final int end;
}
