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

import java.util.IdentityHashMap;
import java.util.Map;

import org.tensorflow.Output;
import org.tensorflow.Tensor;

import com.kubx.tensorflow.data.index.IntIndex;
import com.kubx.tensorflow.data.index.LongIndex;

final class IndexMapper {
  
  static Index map(Object obj) {
    // First check if there is a index creator registered for this kind of object
    IndexCreator indexCreator = indexCreators.get(obj.getClass());
    if (indexCreator != null) {
      return indexCreator.create(obj);
    }
    // Else, assume that the object is an index itself or report an error
    try {
      return (Index)obj;
    } catch (ClassCastException e) {
      throw new IllegalArgumentException("Cannot use object of type \"" + obj.getClass().getName() + "\" as an index to a tensor value");
    }
  }
  
  private interface IndexCreator {
    Index create(Object obj);
  }
  
  private static Map<Class<?>, IndexCreator> indexCreators = new IdentityHashMap<>();
  static {
    indexCreators.put(Tensor.class, new IndexCreator(){
        @Override
        @SuppressWarnings("rawtypes")
        public Index create(Object obj) {
          return tensorToIndex((Tensor)obj);
        }
    });
    indexCreators.put(Output.class, new IndexCreator(){
        @Override
        public Index create(Object obj) {
          return tensorToIndex(((Output<?>)obj).tensor());
        }
    });
    indexCreators.put(IntValue.class, new IndexCreator(){
        @Override
        public Index create(Object obj) {
          return new IntIndex((IntValue)obj);
        }
    });
    indexCreators.put(LongValue.class, new IndexCreator(){
        @Override
        public Index create(Object obj) {
          return new LongIndex((LongValue)obj);
        }
    });
  }
  
  @SuppressWarnings("unchecked")
  private static Index tensorToIndex(Tensor<?> tensor) {
    switch (tensor.dataType()) {
    case INT32:
      return new IntIndex(IntValue.of((Tensor<Integer>)tensor));
    case INT64:
      return new LongIndex(LongValue.of((Tensor<Long>)tensor));
    default:
      throw new IllegalArgumentException("Cannot use tensor of datatype " + tensor.dataType() + " as an index vector");
    }
  }
  
  private IndexMapper() {}
}
