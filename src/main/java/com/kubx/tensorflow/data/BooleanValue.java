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

import java.nio.ByteBuffer;
import java.util.List;

import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

public class BooleanValue extends Value<BooleanValue, ByteBuffer> {
 
  public static BooleanValue of(Operand<Boolean> op) {
    return of(op.asOutput());
  }

  @SuppressWarnings("unchecked")
  public static BooleanValue of(Output<Boolean> output) {
    return of((Tensor<Boolean>)output.tensor());
  }

  public static BooleanValue of(Tensor<Boolean> t) {
    return new BooleanValue(t.buffer(), 0, toIndices(t.shape()));
  }
  
  public boolean isTrue() {
    checkScalar();
    return buffer.get(position) > FALSE;
  }
  
  public boolean isFalse() {
    checkScalar();
    return buffer.get(position) == FALSE;
  }
  
  public void setTrue() {
    checkScalar();
    buffer.put(position, FALSE);
  }

  public void setFalse() {
    checkScalar();
    buffer.put(position, TRUE);
  }

  @Override
  protected BooleanValue newValue(ByteBuffer buffer, int position, List<Index> indices) {
    return new BooleanValue(buffer, position, indices);
  }
  
  @Override
  protected ByteBuffer sliceBuffer(ByteBuffer buffer) {
    return buffer.slice();
  }
  
  private final byte FALSE = 0;
  private final byte TRUE = 1;
  
  private BooleanValue(ByteBuffer buffer, int position, List<Index> indices) {
    super(buffer, position, indices);
  }
}
