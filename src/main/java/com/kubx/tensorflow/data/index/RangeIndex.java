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

public class RangeIndex extends Index {
  
  @Override
  public int size() {
    return end - start;
  }

  @Override
  protected int position(int index) {
    return start + index;
  }  
  
  @Override
  protected boolean vectorizable() {
    return true;
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
