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
