import openneuro as on

if __name__ == '__main__':
    dsid = 'ds003400'
    bids_dir = f'/Users/adam2392/OneDrive - Johns Hopkins/{dsid}/'
    on.download(dataset=dsid, target_dir=bids_dir, exclude='sourcedata')