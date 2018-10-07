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

import java.nio.FloatBuffer;
import java.util.List;

import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

public class FloatValue extends Value<FloatValue, FloatBuffer> {
 
  public static FloatValue of(Operand<Float> op) {
    return of(op.asOutput());
  }

  @SuppressWarnings("unchecked")
  public static FloatValue of(Output<Float> output) {
    return of((Tensor<Float>)output.tensor());
  }

  public static FloatValue of(Tensor<Float> t) {
    return new FloatValue(t.buffer().asFloatBuffer(), 0, toIndices(t.shape()));
  }
  
  public float scalar() {
    checkScalar();
    return buffer.get(position);
  }
  
  public void scalar(float scalar) {
    checkScalar();
    buffer.put(position, scalar);
  }
  
  @Override
  protected FloatValue newValue(FloatBuffer buffer, int position, List<Index> indices) {
    return new FloatValue(buffer, position, indices);
  }
  
  @Override
  protected FloatBuffer sliceBuffer(FloatBuffer buffer) {
    return buffer.slice();
  }
  
  private FloatValue(FloatBuffer buffer, int position, List<Index> indices) {
    super(buffer, position, indices);
  }
}
