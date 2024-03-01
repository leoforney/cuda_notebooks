import java.util.Arrays;

public class ListSort {
    static {
        System.load(System.getProperty("user.dir") + "/../cmake-build-debug/liblist_sort.so");
    }

    public native float[] sort(float[] nums);

    public static void main(String[] args) {
        ListSort obj = new ListSort();
        float[] arr = {5.645f, 9.45687f, 15.001f, 236.57f, 96.54f, 0.0f, -2.45f, 163.56998f, 26.4142f, 94.12f, 20004.78f};
        float[] result = obj.sort(arr);
        System.out.println(Arrays.toString(result));
    }
}