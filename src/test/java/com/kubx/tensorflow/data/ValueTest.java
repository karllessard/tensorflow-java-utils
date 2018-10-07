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

import static com.kubx.tensorflow.data.index.TFIndex.FULL;
import static com.kubx.tensorflow.data.index.TFIndex.range;
import static com.kubx.tensorflow.data.index.TFIndex.sequence;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.nio.IntBuffer;

import org.junit.Test;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

public class ValueTest {
  
  @Test
  public void getScalar() {
    int scalar = 10;
    try (Tensor<Integer> t = Tensors.create(scalar)) {
      IntValue value = IntValue.of(t);
      assertEquals(1L, value.size());
      assertEquals(scalar, value.scalar());
    }
  }
  
  @Test
  public void setScalar() {
    int scalar = 10;
    try (Tensor<Integer> t = Tensors.create(0)) {
      IntValue value = IntValue.of(t);
      value.scalar(scalar);
      assertEquals(scalar, value.scalar());
    }
  }
  
  @Test
  public void cannotIndexScalar() {
    int scalar = 10;
    try (Tensor<Integer> t = Tensors.create(scalar)) {
      IntValue.of(t).at(0);
    } catch (IndexOutOfBoundsException e) {
      System.out.println(e.getMessage());
    }
  }
  
  @Test
  public void cannotConvertVectorToScalar() {
    int[] vector = new int[] {1, 2, 3, 4};
    try (Tensor<Integer> t = Tensors.create(vector)) {
      IntValue.of(t).scalar();
      fail();
    } catch (IllegalArgumentException e) {
      System.out.println(e.getMessage());
    }
  }
  
  @Test
  public void cannotUseStringAsIndexToVector() {
    int[] vector = new int[] {1, 2, 3, 4};
    try (Tensor<Integer> t = Tensors.create(vector)) {
      IntValue.of(t).at("hi there");
      fail();
    } catch (IllegalArgumentException e) {
      System.out.println(e.getMessage());
    }
  }

  @Test
  public void getVector() {
    int[] vector = new int[] {1, 2, 3};
    try (Tensor<Integer> t = Tensors.create(vector)) {
      IntValue value = IntValue.of(t);
      assertEquals(IntBuffer.wrap(vector), value.vector());
    }
  }
  
  @Test
  public void resetVector() {
    int[] originalVector = new int[] {1, 2, 3};
    int[] newVector = new int[] {4, 5, 6};
    try (Tensor<Integer> t = Tensors.create(originalVector)) {
      IntValue value = IntValue.of(t);
      value.vector().put(newVector);
      assertEquals(IntBuffer.wrap(newVector), value.vector());
    }
  }
  
  @Test
  public void getScalarInVector() {
    int[] vector = new int[] {1, 2, 3};
    try (Tensor<Integer> t = Tensors.create(vector)) {
      IntValue value = IntValue.of(t);
      assertEquals(1, value.at(0).scalar());
      assertEquals(2, value.at(1).scalar());
      assertEquals(3, value.at(2).scalar());
    }
  }
  
  @Test
  public void getMatrix() {
    int[][] matrix = new int[][] {
      {1, 2, 3}, {4, 5, 6}
    };
    try (Tensor<Integer> t = Tensors.create(matrix)) {
      IntValue value = IntValue.of(t);
      assertEquals(2, value.size());
    }
  }

  @Test
  public void getVectorInMatrix() {
    int[][] matrix = new int[][] {
      {1, 2, 3}, {4, 5, 6}
    };
    try (Tensor<Integer> t = Tensors.create(matrix)) {
      IntValue value = IntValue.of(t).at(1);
      assertEquals(3, value.size());
      assertEquals(IntBuffer.wrap(matrix[1]), value.vector());
    }
  }

  @Test
  public void getScalarInMatrix() {
    int[][] matrix = new int[][] {
      {1, 2, 3}, {4, 5, 6}
    };
    try (Tensor<Integer> t = Tensors.create(matrix)) {
      IntValue value = IntValue.of(t).at(1, 2);
      assertEquals(6, value.scalar());
    }
  }

  @Test
  public void getSecondAndThirdVectorsInMatrix() {
    int[][] matrix = new int[][] {
      {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}
    };
    try (Tensor<Integer> t = Tensors.create(matrix)) {
      IntValue value = IntValue.of(t).at(range(1, 3));
      assertEquals(2, value.size());
      IntValue vector1 = value.at(0);
      assertEquals(3, vector1.size());
      assertEquals(IntBuffer.wrap(matrix[1]), vector1.vector());
      IntValue vector2 = value.at(1);
      assertEquals(3, vector2.size());
      assertEquals(IntBuffer.wrap(matrix[2]), vector2.vector());
    }
  }
  
  @Test
  public void getSecondAndThirdScalarsOfSecondVectorInMatrix() {
    int[][] matrix = new int[][] {
      {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}
    };
    try (Tensor<Integer> t = Tensors.create(matrix)) {
      IntValue value = IntValue.of(t).at(1, range(1, 3));
      assertEquals(2, value.size());
      assertEquals(5, value.at(0).scalar());
      assertEquals(6, value.at(1).scalar());
    }
  }

  @Test
  public void getThirdScalarOfSecondAndThirdVectorInMatrix() {
    int[][] matrix = new int[][] {
      {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}
    };
    try (Tensor<Integer> t = Tensors.create(matrix)) {
      IntValue value = IntValue.of(t).at(range(1, 3), 2);
      assertEquals(2, value.size());
      IntValue scalar1 = value.at(0);
      assertEquals(1, scalar1.size());
      assertEquals(6, scalar1.scalar());
      IntValue scalar2 = value.at(1);
      assertEquals(1, scalar2.size());
      assertEquals(9, scalar2.scalar());
    }
  }

  @Test
  public void getSecondAndThirdScalarsOfSecondAndThirdVectorsInMatrix() {
    int[][] matrix = new int[][] {
      {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}
    };
    try (Tensor<Integer> t = Tensors.create(matrix)) {
      IntValue value = IntValue.of(t).at(range(1, 3), range(1, 3));
      assertEquals(2, value.size());
      IntValue vector1 = value.at(0);
      assertEquals(2, vector1.size());
      assertEquals(5, vector1.at(0).scalar());
      assertEquals(6, vector1.at(1).scalar());
      IntValue vector2 = value.at(1);
      assertEquals(2, vector2.size());
      assertEquals(8, vector2.at(0).scalar());
      assertEquals(9, vector2.at(1).scalar());
    }
  }
  
  @Test
  public void getAllScalarsInVector() {
    int[] vector = new int[] {1, 2};
    try (Tensor<Integer> t = Tensors.create(vector)) {
      IntValue value = IntValue.of(t).at(FULL);
      assertEquals(2, value.size());
      assertEquals(1, value.at(0).scalar());
      assertEquals(2, value.at(1).scalar());
    }
  }
  
  @Test
  public void getLastScalarOfAllVectorsInMatrix() {
    int[][] matrix = new int[][] {
      {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}
    };
    try (Tensor<Integer> t = Tensors.create(matrix)) {
      IntValue value = IntValue.of(t).at(FULL, 2);
      assertEquals(4, value.size());
      assertEquals(3, value.at(0).scalar());
      assertEquals(6, value.at(1).scalar());
      assertEquals(9, value.at(2).scalar());
      assertEquals(12, value.at(3).scalar());
    }
  }

  @Test
  public void getScalarsOfLastVectorsOfSecondDimensionInMatrix3d() {
    int[][][] matrix = new int[][][] {
      {
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9}
      },
      { 
        {10, 11, 12}, {13, 14, 15}, {16, 17, 18}
      }
    };
    try (Tensor<Integer> t = Tensors.create(matrix)) {
      IntValue value = IntValue.of(t).at(FULL, 2, 2);
      assertEquals(2, value.size());
      assertEquals(9, value.at(0).scalar());
      assertEquals(18, value.at(1).scalar());
    }
  }
  
  @Test
  public void getSubRangeOnTopOfRangeIndexInVector() {
    int[] vector = new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    try (Tensor<Integer> t = Tensors.create(vector)) {
      IntValue value = IntValue.of(t).at(range(3, 9)).at(range(2, 4));
      assertEquals(2, value.size());
      assertEquals(6, value.at(0).scalar());
      assertEquals(7, value.at(1).scalar());
    }
  }
  
  @Test
  public void getSecondAndFourthScalarsInVector() {
    int[] vector = new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    try (Tensor<Integer> t = Tensors.create(vector)) {
      IntValue value = IntValue.of(t).at(sequence(1, 3));
      assertEquals(2, value.size());
      assertEquals(2, value.at(0).scalar());
      assertEquals(4, value.at(1).scalar());
    }
  }

  @Test
  public void getFourthAndSecondScalarsInVector() {
    int[] vector = new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    try (Tensor<Integer> t = Tensors.create(vector)) {
      IntValue value = IntValue.of(t).at(sequence(3, 1));
      assertEquals(2, value.size());
      assertEquals(4, value.at(0).scalar());
      assertEquals(2, value.at(1).scalar());
    }
  }
 
  @Test
  public void getScalarsInVectorFromSparseTensor() {
    int[] vector = new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int[] index = new int[] {2, 4, 6};
    try (Tensor<Integer> t = Tensors.create(vector); 
        Tensor<Integer> idx = Tensors.create(index)) {
      IntValue value = IntValue.of(t).at(idx);
      assertEquals(3, value.size());
      assertEquals(3, value.at(0).scalar());
      assertEquals(5, value.at(1).scalar());
      assertEquals(7, value.at(2).scalar());
    }
  }
  
  @Test
  public void getVectorsAndScalarsInMatrix4d() {
    int[][][][] matrix4d = new int[][][][]{
      {
        {
          {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}
        },
        {
          {13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24}
        }
      },
      {
        {
          {25, 26, 27}, {28, 29, 30}, {31, 32, 33}, {34, 35, 36}
        },
        {
          {37, 38, 39}, {40, 41, 42}, {43, 44, 45}, {46, 47, 48}
        }
      }
    };
    try (Tensor<Integer> t = Tensors.create(matrix4d)) {
      IntValue value = IntValue.of(t);
      assertEquals(2, value.size());
      assertEquals(2, value.at(1).size());
      assertEquals(4, value.at(1, 1).size());
      assertEquals(3, value.at(1, 1, 1).size());
      assertEquals(2, value.at(1, 1, range(2, 4)).size());
      assertEquals(41, value.at(1, 1, 1, 1).scalar());
      assertEquals(IntBuffer.wrap(matrix4d[1][1][3]), value.at(1, 1, 3).vector());
      assertEquals(35, value.at(1, FULL, range(2, 4)).at(0, 1, 1).scalar());
    }
  }
}
