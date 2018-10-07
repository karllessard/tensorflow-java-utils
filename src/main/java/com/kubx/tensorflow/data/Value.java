package com.kubx.tensorflow.data;

import java.nio.Buffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import com.kubx.tensorflow.data.index.OffsetIndex;

abstract class Value<T, U extends Buffer> {
  
  public long size() {
    return indices.isEmpty() ? 1L : indices.get(0).size();
  }
 /* 
  public U vector() {
    // FIXME not all 1-dim values could be tranformed to a vector
    if (dims.size() != 1) {
      throw new IllegalArgumentException("Cannot convert value of " + dims.size() + " dimensions to vector");
    }
    buffer.position(position);
    U vector = slice();
    vector.limit(dims.get(0).size());
    return vector;
  }
 */ 
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

  protected abstract T newValue(U buffer, int position, List<Index> indices);
  protected abstract U slice();

  protected final List<Index> indices;
  protected final U buffer;
  protected final int position;
  
  protected Value(U buffer, int position, List<Index> indices) {
    this.buffer = buffer;
    this.indices = indices;
    this.position = position;
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
}