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
