package com.kubx.tensorflow.data.index;

import com.kubx.tensorflow.data.Index;

public class CombinedIndex extends Index {

  @Override
  public int size() {
    return frontIndex.size();
  }

  @Override
  public int position(int i) {
    return backIndex.position(frontIndex.position(i));
  }

  public CombinedIndex(Index backIndex, Index frontIndex) {
    this.backIndex = backIndex;
    this.frontIndex = frontIndex;
  }
  
  private Index backIndex;
  private Index frontIndex;
}
