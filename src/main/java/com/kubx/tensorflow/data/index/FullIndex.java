package com.kubx.tensorflow.data.index;

import com.kubx.tensorflow.data.Index;

public class FullIndex extends Index {
  
  @Override
  public int size() {
    throw new UnsupportedOperationException();
  }

  @Override
  public int position(int i) {
    throw new UnsupportedOperationException();
  }
  
  @Override
  protected Index merge(Index previousIndex) {
    return previousIndex;
  }
  
  static final Index INSTANCE = new FullIndex();
}
