package com.kubx.tensorflow.data.index;

import com.kubx.tensorflow.data.IntValue;
import com.kubx.tensorflow.data.Index;

public class IntIndex extends Index {

  @Override
  public int size() {
    return (int)value.size();
  }

  @Override
  public int position(int i) {
    return value.at(i).scalar();
  }
  
  public IntIndex(IntValue value) {
    this.value = value;
  }
 
  private IntValue value;
}
