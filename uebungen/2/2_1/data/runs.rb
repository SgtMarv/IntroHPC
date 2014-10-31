#! /home/max/.rvm/rubies/ruby-1.9.2-p290/bin/ruby

write_file = File.open('runs.dat', 'wb')
[1,5,10,50,100,1000,10000].each do |i|
    avg = 0.0
    err = 0.0
    lines = File.open("runs_#{i.to_s}.dat").readlines
    lines.each do |line|
        unless line.split.count >2
            avg += line.split[1].to_f
        end
    end
    avg = (avg/(lines.count/2.0))
    lines.each do |line|
        unless line.split.count >2
            err += (line.split[1].to_f-avg)**2
        end
    end
    err = Math.sqrt((1/(lines.count/2.0-1))*err)
    write_file << i.to_s + " " + (avg).to_s + " " + (err).to_s + "\n"
end
write_file.close
