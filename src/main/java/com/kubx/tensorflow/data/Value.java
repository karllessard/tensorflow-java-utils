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

import java.nio.Buffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import com.kubx.tensorflow.data.index.OffsetIndex;

abstract class Value<T, U extends Buffer> {
  
  public long size() {
    return indices.isEmpty() ? 1L : indices.get(0).size();
  }
  
  public U vector() {
    if (indices.size() != 1) {
      throw new IllegalArgumentException("Cannot convert value of " + indices.size() + " dimensions to vector");
    }
    Index index = indices.get(0);
    // TODO allow any tensor of rank-1 to be converted into a vector, not only "rows" ones.
    // This could be done by having our own implementation of IntBuffer.
    if (!index.vectorizable()) {
      throw new IllegalArgumentException("Current indexation in value does not support data vectorization");
    }
    // We need to slice twice to be thread-safe and preserve the immutability of the current value: 
    // - the first time, we get a copy of the original buffer
    // - the second time, we reset its start position
    U vectorBuffer = sliceBuffer(buffer);
    vectorBuffer.position(position);
    vectorBuffer = sliceBuffer(vectorBuffer);
    vectorBuffer.limit(index.position(index.size()));
    return vectorBuffer;
  }
  
  public T at(Object... indices) {
    if (indices == null) {
      throw new IllegalArgumentException("At least one index should be provided");
    }
    if (indices.length > this.indices.size()) {
      throw new IndexOutOfBoundsException("Number of indices (" + indices.length + 
          ") exceeds the number of available dimensions (" + this.indices.size() + ")");
    }
    int newPosition = position;
    List<Index> newIndices = new ArrayList<>(this.indices.size());
    Iterator<Index> indexIter = this.indices.iterator();
    for (int i = 0; i < indices.length; ++i) {
      Index index = indexIter.next();
      if (indices[i] instanceof Number) {
        int j = ((Number)indices[i]).intValue();
        newPosition += index.position(j);
      } else {
        newIndices.add(IndexMapper.map(indices[i]).merge(index));
      }
    }
    while (indexIter.hasNext()) {
      newIndices.add(indexIter.next());
    } 
    return newValue(buffer, newPosition, newIndices);
  }
  
  protected static List<Index> toIndices(long[] shape) {
    Index[] indices = new Index[shape.length];
    if (shape.length > 0) {
      OffsetIndex index = new OffsetIndex((int)shape[shape.length - 1], 1);
      indices[indices.length - 1] = index;
      for (int i = indices.length - 2; i >= 0; --i) {
        index = new OffsetIndex((int)shape[i], index.offset() * index.size());
        indices[i] = index;
      }
    }
    return Arrays.asList(indices);
  }
  
  protected final U buffer;
  protected final int position;
  
  protected Value(U buffer, int position, List<Index> indices) {
    this.buffer = buffer;
    this.position = position;
    this.indices = Collections.unmodifiableList(indices);
  }

  protected abstract T newValue(U buffer, int position, List<Index> indices);
  protected abstract U sliceBuffer(U buffer);

  protected void checkScalar() { 
    if (indices.size() > 0) {
      throw new IllegalArgumentException("Cannot convert value of " + indices.size() + " dimensions to scalar");
    }
  }

  private final List<Index> indices;
}