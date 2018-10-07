package com.kubx.tensorflow.data.index;

import com.kubx.tensorflow.data.LongValue;
import com.kubx.tensorflow.data.Index;

public class LongIndex extends Index {

  @Override
  public int size() {
    return (int)value.size();
  }

  @Override
  public int position(int i) {
    return ((Long)value.at(i).scalar()).intValue();
  }
  
  public LongIndex(LongValue value) {
    this.value = value;
  }
 
  private LongValue value;
}
