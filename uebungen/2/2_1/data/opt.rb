#! /home/max/.rvm/rubies/ruby-1.9.2-p290/bin/ruby

write_file = File.open('opt_d.dat', 'wb')
lines = File.open("opt.dat").readlines

avg = 0.0
lines.each do |line|
    unless line.split.count>2
        avg += line.split[1].to_f
        write_file << line.split[1].to_f.to_s + "\n"
    end
end
avg = (avg/(lines.count/2.0))

err = 0.0
lines.each do |line|
    unless line.split.count>2
        err += (line.split[1].to_f-avg)**2
    end
end

err = Math.sqrt((1/(lines.count/2.0-1))*err)

puts avg.to_s + " " + err.to_s + "\n"

write_file.close
