import java.util.Arrays;

public class ListSort {
    static {
        System.load("/tmp/project/cmake-build-debug-docker/liblist_sort.so");
    }

    public native int[] sort(int[] nums);

    public static void main(String[] args) {
        ListSort obj = new ListSort();
        int[] arr = {5, 9, 15, 236, 96, 0, 2, 163, 26, 94, 20004};
        int[] result = obj.sort(arr);
        System.out.println(Arrays.toString(result));
    }
}