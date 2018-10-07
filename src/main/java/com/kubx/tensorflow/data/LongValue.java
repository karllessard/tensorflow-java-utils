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

import java.nio.LongBuffer;
import java.util.List;

import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

public class LongValue extends Value<LongValue, LongBuffer> {
 
  public static LongValue of(Operand<Long> op) {
    return of(op.asOutput());
  }

  @SuppressWarnings("unchecked")
  public static LongValue of(Output<Long> output) {
    return of((Tensor<Long>)output.tensor());
  }

  public static LongValue of(Tensor<Long> t) {
    return new LongValue(t.buffer().asLongBuffer(), 0, toIndices(t.shape()));
  }
  
  public long scalar() {
    checkScalar();
    return buffer.get(position);
  }

  public void scalar(long scalar) {
    checkScalar();
    buffer.put(position, scalar);
  }
  
  @Override
  protected LongValue newValue(LongBuffer buffer, int position, List<Index> indices) {
    return new LongValue(buffer, position, indices);
  }
  
  @Override
  protected LongBuffer sliceBuffer(LongBuffer buffer) {
    return buffer.slice();
  }
  
  private LongValue(LongBuffer buffer, int position, List<Index> indices) {
    super(buffer, position, indices);
  }
}
