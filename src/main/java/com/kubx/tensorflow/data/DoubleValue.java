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

import java.nio.DoubleBuffer;
import java.util.List;

import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

public class DoubleValue extends Value<DoubleValue, DoubleBuffer> {
 
  public static DoubleValue of(Operand<Double> op) {
    return of(op.asOutput());
  }

  @SuppressWarnings("unchecked")
  public static DoubleValue of(Output<Double> output) {
    return of((Tensor<Double>)output.tensor());
  }

  public static DoubleValue of(Tensor<Double> t) {
    return new DoubleValue(t.buffer().asDoubleBuffer(), 0, toIndices(t.shape()));
  }
  
  public double scalar() {
    checkScalar();
    return buffer.get(position);
  }
  
  public void scalar(double scalar) {
    checkScalar();
    buffer.put(position, scalar);
  }
  
  @Override
  protected DoubleValue newValue(DoubleBuffer buffer, int position, List<Index> indices) {
    return new DoubleValue(buffer, position, indices);
  }
  
  @Override
  protected DoubleBuffer sliceBuffer(DoubleBuffer buffer) {
    return buffer.slice();
  }
  
  private DoubleValue(DoubleBuffer buffer, int position, List<Index> indices) {
    super(buffer, position, indices);
  }
}
