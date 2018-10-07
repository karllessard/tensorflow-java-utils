package com.kubx.tensorflow.data.index;

import com.kubx.tensorflow.data.Index;
import com.kubx.tensorflow.data.IntValue;

public class IntIndex extends Index {

  @Override
  public int size() {
    return (int)value.size();
  }

  @Override
  protected int position(int i) {
    return value.at(i).scalar();
  }
  
  public IntIndex(IntValue value) {
    this.value = value;
  }
 
  private IntValue value;
}
