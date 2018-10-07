package com.kubx.tensorflow.data;

import com.kubx.tensorflow.data.index.CombinedIndex;

public abstract class Index {
  
  public abstract int size();
  
  public abstract int position(int i);
  
  protected Index merge(Index previousIndex) {
    return new CombinedIndex(previousIndex, this);
  }
}
