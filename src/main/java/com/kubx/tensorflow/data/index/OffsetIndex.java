package com.kubx.tensorflow.data.index;

import com.kubx.tensorflow.data.Index;

public class OffsetIndex extends Index {
  
  @Override
  public int size() {
    return size;
  }

  @Override
  protected int position(int index) {
    return offset * index;
  }

  @Override
  protected boolean vectorizable() {
    return offset == 1;
  }
  
  public int offset() {
    return offset;
  }
  
  public OffsetIndex(int size, int offset) {
    this.size = size;
    this.offset = offset;
  }
  
  private final int size;
  private final int offset;
}
