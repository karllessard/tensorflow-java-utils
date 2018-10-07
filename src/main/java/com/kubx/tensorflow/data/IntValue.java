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
