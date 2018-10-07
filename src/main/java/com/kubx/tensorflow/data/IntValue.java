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

import java.nio.IntBuffer;
import java.util.List;

import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

public class IntValue extends Value<IntValue, IntBuffer> {
 
  public static IntValue of(Operand<Integer> op) {
    return of(op.asOutput());
  }

  @SuppressWarnings("unchecked")
  public static IntValue of(Output<Integer> output) {
    return of((Tensor<Integer>)output.tensor());
  }

  public static IntValue of(Tensor<Integer> t) {
    return new IntValue(t.buffer().asIntBuffer(), 0, toIndices(t.shape()));
  }
  
  public int scalar() {
    checkScalar();
    return buffer.get(position);
  }

  public void scalar(int scalar) {
    checkScalar();
    buffer.put(position, scalar);
  }
  
  @Override
  protected IntValue newValue(IntBuffer buffer, int position, List<Index> indices) {
    return new IntValue(buffer, position, indices);
  }
  
  @Override
  protected IntBuffer sliceBuffer(IntBuffer buffer) {
    return buffer.slice();
  }
  
  private IntValue(IntBuffer buffer, int position, List<Index> indices) {
    super(buffer, position, indices);
  }
}
