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
package com.kubx.tensorflow.data;

public abstract class Index {
  
  protected static class Combined extends Index {

    @Override
    public int size() {
      return frontIndex.size();
    }

    @Override
    protected int position(int i) {
      return backIndex.position(frontIndex.position(i));
    }

    @Override
    protected boolean vectorizable() {
      return backIndex.vectorizable() && frontIndex.vectorizable();
    }

    protected Combined(Index backIndex, Index frontIndex) {
      this.backIndex = backIndex;
      this.frontIndex = frontIndex;
    }
    
    private Index backIndex;
    private Index frontIndex;
  }
  
  public abstract int size();
  
  protected abstract int position(int i);
  
  protected boolean vectorizable() {
    return false;
  }
  
  protected Index merge(Index previousIndex) {
    return new Combined(previousIndex, this);
  }
}
