/* Copyright 2018 KubX. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package com.kubx.tensorflow.data.index;

import com.kubx.tensorflow.data.Index;
import com.kubx.tensorflow.data.LongValue;

public class LongIndex extends Index {

  @Override
  public int size() {
    return (int)value.size();
  }

  @Override
  protected int position(int i) {
    return ((Long)value.at(i).scalar()).intValue();
  }
  
  public LongIndex(LongValue value) {
    this.value = value;
  }
 
  private LongValue value;
}
