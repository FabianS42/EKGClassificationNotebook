import numpy as np
import einops as eo
from scipy.signal import resample



def time_series_segmentation(data, segment_size, fill_up_value=0, overlapping_segments:int= 0):
    """Splits a time series into segments.
    
    Args:
        data (np.ndarray): Input time series data.
        segment_size (int): Size of each segment.
        fill_up_value (int, optional): Value used to fill up the sequence size if it's not divisible by segment_size. Defaults to 0.
        overlapping_segments (int, optional): Number of overlapping segments. Defaults to 0.

    Returns:
        np.ndarray: Segmented time series data.


    Info:
        2 Dimensions:
        Split a time series int segments input. (batchsize, sequenz size) -> out :(batchsize, sequenz size/segment_size, segment_size)
        E.g. segment_size = 10: in=(batchsize, 200) -> out=(batchsize, 20, 10)
        if sequenz size is not dividable by segment_size. Sequenz size is filed up with fill_up_value
        3 Dimensions:
        Split a time series int segments input. (batchsize, sequenz size, number of variables) -> out :(batchsize, sequenz size/segment_size, number of variables*segment_size)
        E.g. segment_size = 10: in=(batchsize, 200, 5) -> out=(batchsize, 20, 50)
        if sequenz size is not dividable by segment_size. Sequenz size is filed up with fill_up_value

        Overlapping:
        overlapping_segments= 1 -> Output size is in=(batchsize, 200) -> out=(batchsize, (20 * 2)-1, 10) with segment_size = 10: 
        Overlapping : overlapping_segments= 1 
        __00__ __01__ __02__ __03__ __04__  -> __00__ __10__  __01__ __11__ ... __13__ __04__
                ^      ^      ^      ^
            __10__ __11__ __12__ __13__ 

        Overlapping : overlapping_segments= 2
        __00__ __01__ __02__ __03__ __04__  -> __00__ __10__ __20__  __01__ __11__ __21__ ...  __03__ __13__ __23__   __04__
           __10__ __11__ __12__ __13__ 
               __20__ __21__ __22__ __23__ 
    """
    data=np.array(data)

    if len(data.shape) == 2:

        if overlapping_segments != 0:

            new_data = []
            overlapping_segments += 1
            shape = data.shape

            fillup = np.zeros((shape[0], segment_size-shape[1]%segment_size)) + fill_up_value
            input_data_filled_up = np.append(data, fillup, axis=-1)

            batch_size = input_data_filled_up.shape[0]
            seq_size = input_data_filled_up.shape[1]
            new_seq_size = seq_size//segment_size

            new_data += [np.array(np.reshape(input_data_filled_up,(batch_size, new_seq_size, segment_size)))]
            print("shape original segmentation seq:",(new_data[0]).shape)
            data=np.zeros((batch_size,(new_seq_size*overlapping_segments)-(overlapping_segments-1),segment_size))

            shift = segment_size//(overlapping_segments)
            rest=0
           
            if segment_size%overlapping_segments:
                rest = 1

            for i in range(overlapping_segments-1):
               
                cropped = input_data_filled_up[:,shift*(i+1):-(segment_size - (shift*(i+1))+rest)]
                new_data += [np.array(np.reshape(cropped,(batch_size, (new_seq_size-1), segment_size)))]

                print("shape of overlapping seq", i, ":",(new_data[i+1]).shape)

            for batch in range(batch_size):
                for seq in range(new_seq_size-1):
                    for overlap in range(overlapping_segments):
                            data[batch,seq+overlap,:] = new_data[overlap][batch,seq,:]

                data[batch,seq+overlap,: ] = new_data[0][batch,-1,:]

            print("shape of result: ", np.array(data).shape)
            return np.array(data)   

        else:
            shape = data.shape
            fillup = np.zeros((shape[0], segment_size-shape[1]%segment_size)) + fill_up_value
            data = np.append(data, fillup, axis=-1)
            data = np.array(np.reshape(data,(data.shape[0], data.shape[1]//segment_size, segment_size)))
            return data

    elif len(data.shape) == 3:

        if overlapping_segments != 0:

            raise "Not yet implemented "

        else:
            shape = data.shape
            fillup = np.zeros((shape[0], segment_size-shape[1]%segment_size, shape[2])) + fill_up_value
            data = np.append(data, fillup, axis=1)
            data = np.array(np.reshape(data,(data.shape[0], data.shape[1]//segment_size, data.shape[2]*segment_size)))
            return data
    else: 

        print("Input dim error of add_time_stamp")
        result = None


    return result

    


#2D Data: input shape = (number of variablen, sequenz size) Output shape = (number of variablen+1, sequenz size)  
#3D Data: input shape = (batchsize, sequenz size, number of variablen) Output shape = (batchsize, sequenz size, number of variablen+1)  
def add_time_stamp(data, lower_value = -1):
    """Adds a time stamp to the input data.

    Args:
        data (np.ndarray): Input data.
        lower_value (int, optional): Lower bound of the time stamp. Defaults to -1.

    Returns:
        np.ndarray: Data with the time stamp added.

    Info:
        #2D Data: input shape = (number of variablen, sequenz size) Output shape = (number of variablen+1, sequenz size)  
        #3D Data: input shape = (batchsize, sequenz size, number of variablen) Output shape = (batchsize, sequenz size, number of variablen+1) 
    """

    data = np.array(data)

    if len(data.shape) == 2:

        length = data.shape[0]
        
        Position = np.linspace(lower_value,1,length)
        Position = Position[:, np.newaxis]

        result = np.concatenate(( data[:,:],Position[:,:] ), axis=-1)

    elif len(data.shape) == 3:

        length = data.shape[1]
        batch_size= data.shape[0]
        Position = np.linspace(lower_value,1,length)
        Position = Position[np.newaxis,:]
        Position = eo.repeat(Position, '1 d -> batch d', batch = batch_size)

        result = np.concatenate(( data,Position[:,:,np.newaxis] ), axis=-1)
            
    else: 

        print("Input dim error of add_time_stamp")
        result = None

    return result



def resample_timeseries(data, downsample_factor, axis=0):
    """Resamples the time series data along the specified axis.

    Args:
        data (np.ndarray): Input time series data.
        downsample_factor (int): Factor by which to downsample the data.
        axis (int, optional): Axis along which to perform resampling. Defaults to 0.

    Returns:
        np.ndarray: Resampled time series data.
    """

    original_shape = data.shape

    if len(original_shape)==1:
        new_size=int(len(data)/downsample_factor)
        return resample(data, int(new_size))

    elif len(original_shape)==2:
        transposed=False
        if axis==0:
            data = data.T
            transposed=True

        resampled_array=[]
        new_size=int(len(data[0])/downsample_factor)
        for i in range(data.shape[0]):        
            resampled_array += [resample(data[i,:], int(new_size))] 

        if transposed==True:
            resampled_array=np.array(resampled_array).T

        return np.array(resampled_array)
    
    elif len(original_shape)>2:
        #transpose resample axis at last position
        transpose_index=[]
        for i in range(len(original_shape)):
            if i != axis:
                transpose_index += [i]
        transpose_index+=[axis]     
        print(transpose_index)

        data=np.transpose(np.array(data),transpose_index)
        transposed_shape = data.shape


        #reduce to 2 dimesions
        new_size=1
        for i in range(len(data.shape)-1):
            new_size*=data.shape[i]
        data = np.reshape(data, newshape=(new_size, data.shape[-1]))
        

        #resample
        new_size=int(len(data[0])/downsample_factor)
        resampled_array=[]
        for i in range(data.shape[0]):        
            resampled_array += [resample(data[i,:], int(new_size))] 
        resampled_array = np.array(resampled_array)
        

        #reverse Reshape
        new_shape=list(transposed_shape)
        new_shape[-1]=new_size
        resampled_array = np.reshape(resampled_array, newshape=new_shape)

        #reverse transpose
        resampled_array=np.transpose(resampled_array, transpose_index)

    return resampled_array



def mask_time_series_sections(data, relativ_mask_lenght = [0.20]):
    """Masks sections of the time series data.
    Args:
        data (np.ndarray): Input time series data.
        relativ_mask_lenght (list, optional): List of relative mask lengths. Defaults to [0.20].

    Returns:
        np.ndarray: Time series data with masked sections.
    """
    
    data = np.array(data)

    if len(data.shape) == 3:
        seq_len = len(data[0,:,0])        
        for d in range(len(data)):
            if d == 0: #Data Vaildation Example
               data[d,60:60+37,:] = 0
               continue            
            if np.random.random()>0.15:   #15% no masking
                used_mask=np.random.randint(0,len(relativ_mask_lenght))
                for i in range(used_mask+1): 
                    used_relativ_mask_lenght = relativ_mask_lenght[used_mask] * (1 +  (np.random.random()-0.5)/ 5  )  #+- 10% Mask size
                    mask_start = int( np.random.randint(0, int(seq_len-(seq_len*used_relativ_mask_lenght))))
                    mask_stop = int(mask_start + (seq_len*used_relativ_mask_lenght))

                    if np.random.random()<0.15: #15% Random value
                        data[d,mask_start:mask_stop,:]=np.random.random()
                    else:
                        data[d,mask_start:mask_stop,:]=0

    return data   




def to_multiple_series(data, target_lenght_of_splited_timeseries, time_shift_after_each_split = None, verbose=False):
    """Split the time series into multiple smaller series.

    Args:
        data (np.ndarray): Input time series data.
        target_length_of_splited_timeseries (int): Target lengths of each new time series.
        time_shift_after_each_split (int, optional): Shift after each split. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        np.ndarray: Array of multiple smaller time series.
    """
    if len(np.array(data).shape) == 1:

        splited_data = split_1d_timeSeries_to_multiple(data, target_lenght_of_splited_timeseries, time_shift_after_each_split, verbose)

    elif len(np.array(data).shape) == 2:

        if data.shape[0]<data.shape[1]:
            data = np.transpose(data, (1,0))

        splited_data = []
        for i in range(data.shape[1]):
            splited_data += [split_1d_timeSeries_to_multiple(data[:,i], target_lenght_of_splited_timeseries, time_shift_after_each_split, verbose)]

        splited_data=np.array(splited_data)

        splited_data = np.transpose(splited_data, (1,2,0))
    else:
        return 0 

    
    return splited_data




def split_1d_timeSeries_to_multiple(data, target_lenght_of_splited_timeseries, time_shift_after_each_split = None, verbose=False):
    """Split a 1D time series into multiple smaller time series.

    Args:
        data (np.ndarray): Input 1D time series data.
        target_length_of_splited_timeseries (int): Target length of each split time series.
        time_shift_after_each_split (int, optional): Shift after each split. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        np.ndarray: Array of multiple smaller time series.
    """
    data = np.expand_dims(data, axis=1)

    num_shifts = int(target_lenght_of_splited_timeseries/time_shift_after_each_split)
    num_splits = int(len(data)/target_lenght_of_splited_timeseries)

    data_cut = data[0:num_splits*target_lenght_of_splited_timeseries-target_lenght_of_splited_timeseries]                   #first split
    #print(len(data_cut))
    batches = np.reshape(data_cut, (num_splits-1, target_lenght_of_splited_timeseries))

    indx_shift =  time_shift_after_each_split-1

    for i in range(num_shifts-1):#second bis x split
    
        data_cut = data[ indx_shift : int(num_splits*target_lenght_of_splited_timeseries)+indx_shift - target_lenght_of_splited_timeseries]
        #print(len(data_cut))
        input_data_filled_up = np.reshape(data_cut, (num_splits-1, target_lenght_of_splited_timeseries))
        batches = np.append(batches, input_data_filled_up, axis=0)
        indx_shift += time_shift_after_each_split       #Shift
    
    
    i = batches.shape[0]

    if verbose:

        print("Finished.", "Total splits:", i)
    
    return batches



